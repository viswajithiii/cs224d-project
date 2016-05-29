'''
A Bidirectional Reccurent Neural Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
from tensorflow.python.ops.constant_op import constant
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import pickle
import sys

'''
To classify words in a sentence as part of Direct Subjective Expressions (DSEs), Expressive
Subjective Expressions, or Neither (O). If a word is a DSE or an ESE, it is either at the beginning(B) or the inside (I) of a phrase. So there are five classes in total: B_DSE, I_DSE, B_ESE, I_ESE and O.

'''
# Parameters
learning_rate = 0.005
# training_iters = 100000
#training_epochs = 200 #Hyperparameter used in paper
training_epochs = 100
minibatch_sentence_size = 80 #Hyperparameter used in paper
batch_size = 64
display_step = 1

# Network Parameters
#n_input = 28 # MNIST data input (img shape: 28*28)
#n_steps = 28 # timesteps

#Added:
n_input = 300 #Word vector length
n_steps = 10 #Number of timesteps.

#n_hidden = 128 # hidden layer num of features
#n_classes = 10 # MNIST total classes (0-9 digits)

#Added:
n_hidden = 100 #Word vector length
n_classes = 5 #BDSE, IDSE, BESE, IESE, O -- five classes

classes_to_int_dict = {
        'BDSE': 0,
        'IDSE': 1,
        'BESE': 2,
        'IESE': 3,
        'O': 4
    }

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

    # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
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
        y = np.reshape(pre_y, (-1, n_classes))
        yield (x, y)


def load_data(fname, vocab):
    """
    Loads data from our train.txt, test.txt and val.txt.
    Expects one word per line, tab-separated from its true tag.
    Creates an 1-d numpy array with the indexes of the word in the vocabulary,
    and one with the corresponding tags.
    """
    f = open(fname, 'r')
    word_idxs = []
    tags = []
    for line in f:
        word, tag = line.split('\t')
        if tag.endswith('\n'):
            tag = tag[:-1]
        word_idxs.append(vocab.encode(word))
        tags.append(classes_to_int_dict[tag])
    return np.asarray(word_idxs), np.asarray(tags)

"""
if __name__ == "__main__":  # Testing code
    vocab = pickle.load(open('vocab.pickle', 'r'))
    test_x, test_y = load_data('../data/testsmall.txt', vocab)
    for (x, y) in data_iterator(test_x, test_y, 4, 3, 5):
        print 'Next:'
        print 'X:', [[vocab.decode(a) for a in b] for b in x]
        print 'Y:',y
    assert False
"""

"""

def BiRNN(_X, _istate_fw, _istate_bw, _weights, _biases, _batch_size, _seq_len):

    # BiRNN requires to supply sequence_length as [batch_size, int64]
    # Note: Tensorflow 0.6.0 requires BiRNN sequence_length parameter to be set
    # For a better implementation with latest version of tensorflow, check below
    _seq_len = tf.fill([_batch_size], constant(_seq_len, dtype=tf.int64))

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define lstm cells with tensorflow
    # Forward direction cell
    #lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    #Added
    rnn_fw_cell = rnn_cell.RNNCell(n_hidden)


    # Backward direction cell
    #lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)


    #Added
    rnn_bw_cell = rnn_cell.RNNCell(n_hidden)

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs = rnn.bidirectional_rnn(rnn_fw_cell, rnn_bw_cell, _X,
                                            initial_state_fw=_istate_fw,
                                            initial_state_bw=_istate_bw,
                                            sequence_length=_seq_len)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = BiRNN(x, istate_fw, istate_bw, weights, biases, batch_size, n_steps)
"""


# NOTE: The following code is working with current master version of tensorflow
#       BiRNN sequence_length parameter isn't required, so we don't define it
#
def BiRNN(_X, _istate_fw, _istate_bw, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define lstm cells with tensorflow
    # Forward direction cell
    rnn_fw_cell = rnn_cell.BasicRNNCell(n_hidden)
    # Backward direction cell
    rnn_bw_cell = rnn_cell.BasicRNNCell(n_hidden)

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)
#
    # Get lstm cell output
    outputs, final_fw, final_bw = rnn.bidirectional_rnn(rnn_fw_cell, rnn_bw_cell, _X,
                                            initial_state_fw=_istate_fw,
                                            initial_state_bw=_istate_bw)
#
    # Linear activation
    # Get inner loop last output
    toreturn = []
    for o in outputs:
        toreturn.append(tf.matmul(o, _weights['out']) + _biases['out'])
    return toreturn, final_fw, final_bw
    #return [tf.matmul(output, _weights['out']) + _biases['out'] for output in outputs]


# tf Graph input
# The input placeholder will be batch_size x n_steps
input_placeholder = tf.placeholder(tf.int32, [None, n_steps])
#x = tf.placeholder(tf.float32, [None, n_steps, n_input])
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate_fw = tf.placeholder(tf.float32, [None, n_hidden])
istate_bw = tf.placeholder(tf.float32, [None, n_hidden])

# Labels has to be (batch_size*n_steps) x n_classes
labels = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'hidden': tf.Variable(tf.random_normal([n_input, 2*n_hidden])),
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([2*n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# The embedding matrix; we're making it a constant because backproping into
# it would be overfitting.
embedding_matrix = np.load('embedding_matrix.npy')
# Dimensions are (vocab.size x wordvec_dim)
L = tf.constant(embedding_matrix, dtype=tf.float32)

# Now, we have created all our variables and constants; we need to start
# defining our computation graph.


# First, we need to go from the input word indexes to their word vectors.
inputs = tf.nn.embedding_lookup(L, input_placeholder)
# This will be of dimension batch_size x n_steps x word_dim

# Now, the BiRNN will output a list of predictions
# This will be a python list of length n_steps
# Each output will be of dim batch_size x output_dim
list_preds, final_fw, final_bw = BiRNN(inputs, istate_fw, istate_bw, weights, biases)



# We make one 2-D tensor out of preds, of dimension (batch_size*n_steps) x
# n_classes.
packed_preds = tf.transpose(tf.pack(list_preds), perm=[0, 2, 1]) # This is a tensor of dimension batch_size x n_steps x n_classes
preds = tf.reshape(packed_preds, (-1, n_classes))


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(preds, labels))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model
correct_pred = tf.equal(tf.argmax(preds,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Now, we've made our computational graph.
# Let's load the data
vocab = pickle.load(open('vocab.pickle', 'r'))
x_train, y_train = load_data('../data/train.txt', vocab)
x_dev, y_dev = load_data('../data/dev.txt', vocab)
x_test, y_test = load_data('../data/test.txt', vocab)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    verbose = 40
    total_steps = sum(1 for x in data_iterator(x_train, y_train, batch_size, n_steps, n_classes))
    # Keep training until reach max epochs
    while step < training_epochs:
        print 'Epoch number:', step
        curr_istate_fw = np.zeros((batch_size, n_hidden))
        curr_istate_bw = np.zeros((batch_size, n_hidden))
        for (i, (x_batch, y_batch)) in enumerate(data_iterator(x_train, y_train, batch_size, n_steps, n_classes)):

            # Fit training using batch data
            _, curr_loss, curr_istate_fw, curr_istate_bw, acc = sess.run(
                [optimizer, cost, final_fw, final_bw, accuracy],
                                          feed_dict={input_placeholder: x_batch,
                                           labels: y_batch,
                                            istate_fw: curr_istate_fw,
                                            istate_bw: curr_istate_bw})
            if verbose and (i % verbose == 0 or i >= total_steps -1):
                sys.stdout.write('\r{} / {} : current training cost = {}, curr_acc = {}'.format(
                    i, total_steps, curr_loss, acc))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\n')

        if step % display_step == 0:
            # Calculate batch accuracy
            losses = []
            accs = []
            for x_batch, y_batch in data_iterator(x_dev, y_dev, batch_size, n_steps, n_classes):
                acc, loss = sess.run([accuracy, cost], feed_dict={input_placeholder: x_batch,
                                                                    labels: y_batch,
                                                    istate_fw: np.zeros((batch_size, n_hidden)),
                                                    istate_bw: np.zeros((batch_size, n_hidden))})
                accs.append(acc)
                losses.append(loss)
            loss = sum(losses)/len(losses)
            acc = sum(accs)/len(accs)
            print "Validation Loss= " + "{:.6f}".format(loss) + \
                    ", Validation Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    """
    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                             istate_fw: np.zeros((test_len, n_hidden)),
                                                             istate_bw: np.zeros((test_len, n_hidden))})
    """
