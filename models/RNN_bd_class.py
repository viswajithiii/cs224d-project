import tensorflow as tf
# from tensorflow.python.ops.constant_op import constant
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import pickle
import sys

"""
To classify words in a sentence as part of Direct Subjective Expressions (DSEs),
Expressive Subjective Expressions, or Neither (O). If a word is a DSE or an ESE,
it is either at the beginning(B) or the inside (I) of a phrase. So there are
five classes in total: B_DSE, I_DSE, B_ESE, I_ESE and O.
"""

DEBUG = False


class Config(object):
    """
    Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Parameters
    lr = 0.005
    # training_iters = 100000
    # training_epochs = 200 #Hyperparameter used in paper
    training_epochs = 100
    minibatch_sentence_size = 80  # Hyperparameter used in paper
    batch_size = 64
    display_step = 1

    # Added:
    num_input = 300  # Word vector length
    num_steps = 10  # Number of timesteps.

    # n_hidden = 128 # hidden layer num of features
    # n_classes = 10 # MNIST total classes (0-9 digits)

    # Added:
    num_hidden = 100  # Number of hidden states
    num_classes = 5  # BDSE, IDSE, BESE, IESE, O -- five classes

classes_to_int_dict = {
        'BDSE': 0,
        'IDSE': 1,
        'BESE': 2,
        'IESE': 3,
        'O': 4
    }

int_to_classes_dict = {v: k for k, v in classes_to_int_dict.iteritems()}


# UTILITY METHODS

def data_iterator(raw_x, raw_y, batch_size, num_steps, num_classes):
    """
    Returns a generator that can iterate over our data in batches.
    raw_data is taken from the output of load_data.
    raw_x is a 1d array of all the word indexes.
    raw_y is a 1d array of all the corresponding tags.

    returns x and y that can be used by our tensorflow model
    x is batch_size x n_steps
    y is (batch_size*n_steps) x n_classes
    """

    data_len = len(raw_x)
    num_batches = data_len // batch_size

    # We need y as a matrix with one-hot rows instead of a list of indices
    # The following two lines perform this conversion.
    mat_y = np.zeros((data_len, num_classes))
    mat_y[np.arange(data_len), raw_y] = 1

    # Reshape raw_x into an array of batch_size x num_batches
    # and raw_y into an array of batch_size x num_batches x num_classes
    res_x = np.zeros([batch_size, num_batches], dtype=np.int32)
    res_y = np.zeros([batch_size, num_batches, num_classes], dtype=np.int32)
    for i in range(batch_size):
        res_x[i] = raw_x[num_batches * i:num_batches * (i + 1)]
        res_y[i] = mat_y[num_batches*i:num_batches*(i+1), :]
    epoch_size = (num_batches - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = res_x[:, i * num_steps:(i + 1) * num_steps]
        pre_y = res_y[:, i * num_steps:(i + 1) * num_steps]
        y = np.reshape(pre_y, (-1, num_classes))
        yield (x, y)


def print_confusion(confusion, num_to_tag):
    """Helper method that prints confusion matrix."""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print
    print confusion
    for i, tag in sorted(num_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        f1 = 2*prec*recall/(prec+recall)
        print 'Tag: {} - P {:2.4f} / R {:2.4f} / F1 {:2.4f}'.format(
            tag, prec, recall, f1)


def calculate_confusion(config, predicted_indices, y_indices):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((config.num_classes, config.num_classes),
                         dtype=np.int32)
    for i in xrange(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion


class BiRNN_Classifier:

    def load_data(self, debug=False):

        self.embedding_matrix = np.load('embedding_matrix.npy')
        self.vocab = pickle.load(open('vocab.pickle', 'r'))
        self.x_train, self.y_train = self.load_data_from_file(
            '../data/train.txt')
        self.x_dev, self.y_dev = self.load_data_from_file('../data/dev.txt')
        self.x_test, self.y_test = self.load_data_from_file('../data/test.txt')
        if debug:
            self.x_train = self.x_train[:1024]
            self.y_train = self.y_train[:1024]

    def load_data_from_file(self, fname):
        """
        Loads data from our train.txt, test.txt and val.txt.
        Expects one word per line, tab-separated from its true tag.
        Creates an 1-d numpy array with the indexes of the word in the
        vocabulary, and one with the corresponding tags.
        """
        f = open(fname, 'r')
        word_idxs = []
        tags = []
        for line in f:
            word, tag = line.split('\t')
            if tag.endswith('\n'):
                tag = tag[:-1]
            word_idxs.append(self.vocab.encode(word))
            tags.append(classes_to_int_dict[tag])
        return np.asarray(word_idxs), np.asarray(tags)

    def add_placeholders(self):

        # The input placeholder will be batch_size x n_steps
        self.input_placeholder = tf.placeholder(tf.int32,
                                                [None, self.config.num_steps])

        # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
        self.istate_fw = tf.placeholder(tf.float32,
                                        [None, self.config.num_hidden])
        self.istate_bw = tf.placeholder(tf.float32,
                                        [None, self.config.num_hidden])

        # Labels has to be (batch_size*n_steps) x n_classes
        self.labels_placeholder = tf.placeholder(tf.float32,
                                                 [None,
                                                  self.config.num_classes])

    def add_weights(self):

        # Define weights
        self.weights = {
            # Hidden layer weights => 2*n_hidden because of foward + backward
            # cells
            'hidden': tf.get_variable("W_h", shape=(self.config.num_input,
                                                    2*self.config.num_hidden)),
            'out': tf.get_variable("W_o", shape=(2*self.config.num_hidden,
                                                 self.config.num_classes))
        }

        self.biases = {
            'hidden': tf.get_variable("b_h", shape=(2*self.config.num_hidden,)),
            'out': tf.get_variable("b_o", shape=(self.config.num_classes,))
        }

    def add_embedding(self):
        # The embedding matrix; we're making it a constant because backproping
        # into it would be overfitting.
        # Dimensions are (vocab.size x wordvec_dim)
        self.L = tf.constant(self.embedding_matrix, dtype=tf.float32)
        # First, we need to go from the input word indexes to their word vecs.
        inputs = tf.nn.embedding_lookup(self.L, self.input_placeholder)
        # This will be of dimension batch_size x n_steps x word_dim
        return inputs

    def BiRNN(self, _X, _istate_fw, _istate_bw, _weights, _biases):

        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        # (n_steps*batch_size, n_input)
        _X = tf.reshape(_X, [-1, self.config.num_input])
        # Linear activation
        _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

        # Forward direction cell
        rnn_fw_cell = rnn_cell.BasicRNNCell(self.config.num_hidden)
        # Backward direction cell
        rnn_bw_cell = rnn_cell.BasicRNNCell(self.config.num_hidden)

        # Split data because rnn cell needs a list of inputs for the RNN inner
        # loop
        # n_steps * (batch_size, n_hidden)
        _X = tf.split(0, self.config.num_steps, _X)

        # Get lstm cell output
        outputs, final_fw, final_bw = rnn.bidirectional_rnn(
                                        rnn_fw_cell, rnn_bw_cell, _X,
                                        initial_state_fw=_istate_fw,
                                        initial_state_bw=_istate_bw)
        # Linear activation
        return [tf.matmul(output, _weights['out']) + _biases['out']
                for output in outputs], final_fw, final_bw

    def create_computation_graph(self):

        self.add_placeholders()
        self.add_weights()
        inputs = self.add_embedding()

        # Now, the BiRNN will output a list of predictions
        # This will be a python list of length n_steps
        # Each output will be of dim batch_size x output_dim
        list_preds, self.final_fw, self.final_bw = self.BiRNN(
            inputs, self.istate_fw, self.istate_bw, self.weights, self.biases)

        # We make one 2-D tensor out of preds, of dimension (batch_size*n_steps)
        # x n_classes.
        # This is a tensor of dimension batch_size x n_steps x n_classes
        packed_preds = tf.transpose(tf.pack(list_preds), perm=[1, 0, 2])
        preds = tf.reshape(packed_preds, (-1, self.config.num_classes))

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            preds, self.labels_placeholder))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr
                                                ).minimize(self.cost)

        # Evaluate model
        self.predicted_indices = tf.argmax(preds, 1)
        self.true_indices = tf.argmax(self.labels_placeholder, 1)
        correct_pred = tf.equal(self.predicted_indices, self.true_indices)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Now, we've made our computational graph.
    def __init__(self, config):
        self.config = config
        self.load_data(DEBUG)
        self.create_computation_graph()

    def run_epoch(self, sess, verbose=40):

        total_steps = sum(1 for x in data_iterator(
            self.x_train, self.y_train, self.config.batch_size,
            self.config.num_steps, self.config.num_classes))

        curr_istate_fw = np.zeros((self.config.batch_size,
                                   self.config.num_hidden))

        curr_istate_bw = np.zeros((self.config.batch_size,
                                   self.config.num_hidden))

        for (i, (x_batch, y_batch)) in enumerate(
            data_iterator(self.x_train, self.y_train, self.config.batch_size,
                          self.config.num_steps, self.config.num_classes)):

            # Fit training using batch data
            _, curr_loss, curr_istate_fw, curr_istate_bw, acc = sess.run(
                [self.optimizer, self.cost, self.final_fw, self.final_bw,
                 self.accuracy], feed_dict={
                            self.input_placeholder: x_batch,
                            self.labels_placeholder: y_batch,
                            self.istate_fw: curr_istate_fw,
                            self.istate_bw: curr_istate_bw})

            if verbose and (i % verbose == 0 or i >= total_steps - 1):
                sys.stdout.write(
                    '\r{} / {} : current training cost = {}, curr_acc = {}'.
                    format(i, total_steps, curr_loss, acc))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\n')

    def calculate_dev_accuracy(self, sess):
        # Calculate batch accuracy
        losses = []
        accs = []
        curr_istate_fw = np.zeros((self.config.batch_size,
                                   self.config.num_hidden))

        curr_istate_bw = np.zeros((self.config.batch_size,
                                   self.config.num_hidden))

        predicted_indices = []
        true_indices = []
        for x_batch, y_batch in data_iterator(self.x_dev, self.y_dev,
                                              self.config.batch_size,
                                              self.config.num_steps,
                                              self.config.num_classes):
            acc, loss, pred_idxs, y, curr_istate_fw, curr_istate_bw = sess.run(
                [self.accuracy, self.cost, self.predicted_indices,
                 self.true_indices, self.final_fw,
                    self.final_bw], feed_dict={
                    self.input_placeholder: x_batch,
                    self.labels_placeholder: y_batch,
                    self.istate_fw: np.zeros((self.config.batch_size,
                                              self.config.num_hidden)),
                    self.istate_bw: np.zeros((self.config.batch_size,
                                              self.config.num_hidden))})

            accs.append(acc)
            losses.append(loss)
            predicted_indices.append(pred_idxs)
            true_indices.append(y)
        predicted_indices = np.concatenate(predicted_indices)
        true_indices = np.concatenate(true_indices)
        print_confusion(calculate_confusion(self.config, predicted_indices,
                                            true_indices),
                        int_to_classes_dict)
        loss = sum(losses)/len(losses)
        acc = sum(accs)/len(accs)
        print "Validation Loss= " + "{:.6f}".format(loss) + \
              ", Validation Accuracy= " + "{:.5f}".format(acc)


def run_BiRNN():
    """
    Runs the BiRNN on the MPQA dataset.
    """
    # Launch the graph
    config = Config()

    model = BiRNN_Classifier(config)

    # Initializing the variables
    init = tf.initialize_all_variables()

    with tf.Session() as sess:

        sess.run(init)
        verbose = 10

        # Keep training until reach max epochs
        for step in xrange(config.training_epochs):
            model.run_epoch(sess, verbose)
            if step % config.display_step == 0:
                model.calculate_dev_accuracy(sess)

        print "Optimization Finished!"

if __name__ == "__main__":
    run_BiRNN()
