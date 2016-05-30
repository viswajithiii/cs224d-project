import tensorflow as tf
# from tensorflow.python.ops.constant_op import constant
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import pickle
import sys
import time

"""
To classify words in a sentence as part of Direct Subjective Expressions (DSEs),
Expressive Subjective Expressions, or Neither (O). If a word is a DSE or an ESE,
it is either at the beginning(B) or the inside (I) of a phrase. So there are
five classes in total: B_DSE, I_DSE, B_ESE, I_ESE and O.
"""

DEBUG = True

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

    num_input = 300  # Word vector length
    # num_steps = 400  # Number of timesteps.

    num_classes = 5  # BDSE, IDSE, BESE, IESE, O -- five classes

    def __init__(self, nlayers, nhidden, nsteps, input_keep_prob, output_keep_prob):
        self.model_depth = nlayers
        self.num_hidden = nhidden
        self.num_steps = nsteps
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        

classes_to_int_dict = {
        'BDSE': 0,
        'IDSE': 1,
        'BESE': 2,
        'IESE': 3,
        'O': 4
    }

int_to_classes_dict = {v: k for k, v in classes_to_int_dict.iteritems()}


# UTILITY METHODS
def data_iterator_rev(raw_x, raw_y, batch_size, num_steps, num_classes):
    """
    Reversed data iterator.
    Returns a generator that can iterate over our data in batches.
    raw_data is taken from the output of load_data.
    raw_x is a 1d array of all the word indexes.
    raw_y is a 1d array of all the corresponding tags.

    returns x and y that can be used by our tensorflow model
    x is batch_size x n_steps
    y is (batch_size*n_steps) x n_classes
    """

    # TODO: Check if this is necessary
    # Augment our data so that the number of words is an exact multiple
    # Of batch_size*num_steps
    total_elems_per_batch = batch_size*num_steps

    data_len = len(raw_x)
    if data_len % total_elems_per_batch != 0:
        new_len = (data_len // total_elems_per_batch + 1)*total_elems_per_batch
        raw_x = np.concatenate([raw_x, np.zeros((new_len - data_len,),
                                                dtype=np.int32)])
        raw_y = np.concatenate([raw_y, classes_to_int_dict['O'] *
                                np.ones((new_len - data_len,),
                                        dtype=np.int32)])

    data_len = new_len
    assert len(raw_x) == data_len
    assert len(raw_y) == data_len

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
    epoch_size = (num_batches) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size - 1, -1, -1):
        x = res_x[:, i * num_steps:(i + 1) * num_steps]
        pre_y = res_y[:, i * num_steps:(i + 1) * num_steps]
        y = np.reshape(pre_y, (-1, num_classes))
        yield (x, y)


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

    # TODO: Check if this is necessary
    # Augment our data so that the number of words is an exact multiple
    # Of batch_size*num_steps
    total_elems_per_batch = batch_size*num_steps

    data_len = len(raw_x)
    if data_len % total_elems_per_batch != 0:
        new_len = (data_len // total_elems_per_batch + 1)*total_elems_per_batch
        raw_x = np.concatenate([raw_x, np.zeros((new_len - data_len,),
                                                dtype=np.int32)])
        raw_y = np.concatenate([raw_y, classes_to_int_dict['O'] *
                                np.ones((new_len - data_len,),
                                        dtype=np.int32)])

    data_len = new_len
    assert len(raw_x) == data_len
    assert len(raw_y) == data_len

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
    epoch_size = (num_batches) // num_steps
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
                                        [None, 2 * self.config.model_depth *
                                         self.config.num_hidden])
        self.istate_bw = tf.placeholder(tf.float32,
                                        [None, 2 * self.config.model_depth *
                                         self.config.num_hidden])

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
        single_fw_cell = rnn_cell.BasicLSTMCell(self.config.num_hidden)
        single_fw_cell = rnn_cell.DropoutWrapper(single_fw_cell, self.config.input_keep_prob, self.config.output_keep_prob, 0.8)
        rnn_fw_cell = rnn_cell.MultiRNNCell(
            [single_fw_cell]*self.config.model_depth)
        # Backward direction cell
        single_bw_cell = rnn_cell.BasicLSTMCell(self.config.num_hidden)
        single_bw_cell = rnn_cell.DropoutWrapper(single_bw_cell, self.config.input_keep_prob, self.config.output_keep_prob)
        rnn_bw_cell = rnn_cell.MultiRNNCell(
            [single_bw_cell]*self.config.model_depth)

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
    def __init__(self, config, verbose=False):
        self.verbose = verbose
        self.config = config
        if self.verbose:
            print 'Loading data ...'
            start_time = time.time()
        self.load_data(DEBUG)
        if self.verbose:
            end_time = time.time()
            print 'Loaded data. Took %f seconds.' % (end_time - start_time)
            print 'Creating computation graph ...'
            start_time = time.time()
        self.create_computation_graph()
        if self.verbose:
            end_time = time.time()
            print 'Created computation graph. Took %f seconds.' % (
                end_time - start_time)

    def run_epoch(self, sess, verbose=40):

        total_steps = sum(1 for x in data_iterator(
            self.x_train, self.y_train, self.config.batch_size,
            self.config.num_steps, self.config.num_classes))

        curr_istate_fw = np.zeros((self.config.batch_size,
                                   2 * self.config.model_depth *
                                   self.config.num_hidden))

        curr_istate_bw = np.zeros((self.config.batch_size,
                                   2 * self.config.model_depth *
                                   self.config.num_hidden))

        for (i, (x_batch, y_batch)) in enumerate(
            data_iterator(self.x_train, self.y_train, self.config.batch_size,
                          self.config.num_steps, self.config.num_classes)):

            # Fit training using batch data
            _, curr_loss, curr_istate_fw, curr_istate_bw, acc, p, y = sess.run(
                [self.optimizer, self.cost, self.final_fw, self.final_bw,
                 self.accuracy, self.predicted_indices, self.true_indices],
                feed_dict={
                            self.input_placeholder: x_batch,
                            self.labels_placeholder: y_batch,
                            self.istate_fw: curr_istate_fw,
                            self.istate_bw: curr_istate_bw})

            if verbose and (i % verbose == 0 or i >= total_steps - 1):
                sys.stdout.write(
                    '\r{} / {} : Current loss = {}, Current accuracy = {}'.
                    format(i, total_steps, curr_loss, acc))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\n')
            print 'Reverse epoch.'

        for (i, (x_batch, y_batch)) in enumerate(
            data_iterator_rev(self.x_train, self.y_train,
                              self.config.batch_size,
                              self.config.num_steps, self.config.num_classes)):

            # Fit training using batch data
            _, curr_loss, curr_istate_fw, curr_istate_bw, acc, p, y = sess.run(
                [self.optimizer, self.cost, self.final_fw, self.final_bw,
                 self.accuracy, self.predicted_indices, self.true_indices],
                feed_dict={
                            self.input_placeholder: x_batch,
                            self.labels_placeholder: y_batch,
                            self.istate_fw: curr_istate_fw,
                            self.istate_bw: curr_istate_bw})

            if verbose and (i % verbose == 0 or i >= total_steps - 1):
                sys.stdout.write(
                    '\r{} / {} : Current loss = {}, Current accuracy = {}'.
                    format(i, total_steps, curr_loss, acc))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\n')

    def calculate_accuracy(self, sess, data_src='dev'):

        src_to_x_dict = {'train': self.x_train, 'dev': self.x_dev,
                         'test': self.x_test}
        src_to_y_dict = {'train': self.y_train, 'dev': self.y_dev,
                         'test': self.y_test}

        data_x = src_to_x_dict[data_src]
        data_y = src_to_y_dict[data_src]

        # Calculate batch accuracy
        losses = []
        accs = []
        curr_istate_fw = np.zeros((self.config.batch_size,
                                   2 * self.config.model_depth *
                                   self.config.num_hidden))

        curr_istate_bw = np.zeros((self.config.batch_size,
                                   2 * self.config.model_depth *
                                   self.config.num_hidden))

        preds = []
        trues = []
        xes = []
        for x_batch, y_batch in data_iterator_rev(data_x, data_y,
                                                  self.config.batch_size,
                                                  self.config.num_steps,
                                                  self.config.num_classes):
            acc, loss, p, y, curr_istate_fw, curr_istate_bw = sess.run(
                [self.accuracy, self.cost, self.predicted_indices,
                 self.true_indices, self.final_fw,
                    self.final_bw], feed_dict={
                    self.input_placeholder: x_batch,
                    self.labels_placeholder: y_batch,
                    self.istate_fw: curr_istate_fw,
                    self.istate_bw: curr_istate_bw})

        for x_batch, y_batch in data_iterator(data_x, data_y,
                                              self.config.batch_size,
                                              self.config.num_steps,
                                              self.config.num_classes):
            acc, loss, p, y, curr_istate_fw, curr_istate_bw = sess.run(
                [self.accuracy, self.cost, self.predicted_indices,
                 self.true_indices, self.final_fw,
                    self.final_bw], feed_dict={
                    self.input_placeholder: x_batch,
                    self.labels_placeholder: y_batch,
                    self.istate_fw: curr_istate_fw,
                    self.istate_bw: curr_istate_bw})

            accs.append(acc)
            losses.append(loss)
            xes.append(x_batch)
            preds.append(p.reshape(x_batch.shape))
            trues.append(y.reshape(x_batch.shape))

        xes = np.concatenate(xes, 1).reshape((-1, ))
        preds = np.concatenate(preds, 1).reshape((-1, ))
        trues = np.concatenate(trues, 1).reshape((-1, ))

        print_confusion(calculate_confusion(self.config, preds,
                                            trues),
                        int_to_classes_dict)
        loss = sum(losses)/len(losses)
        acc = sum(accs)/len(accs)
        print "Validation Loss = " + "{:.6f}".format(loss) + \
              ", Validation Accuracy = " + "{:.5f}".format(acc)

        return xes, preds, trues


def run_BiRNN(nlayers, nhidden, nsteps, input_keep_prob, output_keep_prob):
    """
    Runs the BiRNN on the MPQA dataset.
    """
    # Launch the graph
    config = Config(nlayers, nhidden, nsteps, input_keep_prob, output_keep_prob)

    model = BiRNN_Classifier(config, verbose=True)

    # Initializing the variables
    init = tf.initialize_all_variables()

    with tf.Session() as sess:

        sess.run(init)
        verbose = 10

        step = 0
        # Keep training until reach max epochs
        while step < (config.training_epochs):
            print 'Epoch number:', (step + 1)
            model.run_epoch(sess, verbose)
            if step % config.display_step == 0:
                model.calculate_accuracy(sess)
            step += 1
            
            #if step == config.training_epochs:
            #    print 'All epochs done, do you want to do more? (Y/N)'
            #    print 'If yes, follow Y with the number of epochs to add.'
            #    try:
            #        response = raw_input()
            #        if response[0] == 'Y':
            #            num_steps_to_add = int(response.split()[1])
            #            config.training_epochs += num_steps_to_add
            #            print 'Success. Total epochs set to %d' % (
            #                config.training_epochs)
            #    except:
            #        print 'Something went wrong with your input.'
            #        print 'Not training anymore.'

        for s in ['train', 'dev', 'test']:
            xes, preds, trues = model.calculate_accuracy(sess, s)
            with open('outputs_%s_LSTM_%d_layers_%d_hidden_%d_steps_%f_ipDO_%f_opDO.txt' % 
                (s, nlayers, nhidden, nsteps, input_keep_prob, output_keep_prob), 'w') as out_f:
                for i in range(len(xes)):
                    out_f.write(model.vocab.decode(xes[i]) + '\t' +
                                int_to_classes_dict[preds[i]] +
                                '\t' + int_to_classes_dict[trues[i]] + '\n')
        print "Optimization Finished!"

if __name__ == "__main__":
    #Extract arguments from call to program
    nlayers = 2 
    nhidden = 50
    nsteps = 80
    input_keep_prob = 0.8
    output_keep_prob = 0.8
    if len(sys.argv) > 1:
        nlayers = int(sys.argv[1])
    if len(sys.argv)>2:        
        nhidden = int(sys.argv[2])
    if len(sys.argv)>3:
        nsteps = int(sys.argv[3])
    if len(sys.argv)>4: 
        input_keep_prob = float(sys.argv[4])
        output_keep_prob = float(sys.argv[5])
    run_BiRNN(nlayers, nhidden, nsteps, input_keep_prob, output_keep_prob)
