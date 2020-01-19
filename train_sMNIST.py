"""
This script trains an LSTM (or an RNN) model to solve the sequential (pixel-wise) MNIST task.
The results of this script are reported as a baseline result in:

["Long short-term memory and learning-to-learn in networks of spiking neurons"](https://arxiv.org/abs/1803.09574)
Guillaume Bellec, Darjan Salaj, Anand Subramoney, Robert Legenstein, Wolfgang Maass


Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).


Script is based on the example script by Aymeric Damien:
https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle a sequence of 784 steps for every sample.
'''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'LSTM', 'LSTM or RNN')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'Number of hidden units in LSTM')
tf.app.flags.DEFINE_float('lr', 1e-3, 'Initial learning rate')
tf.app.flags.DEFINE_integer('training_steps', 36000, 'Number of training steps')
tf.app.flags.DEFINE_integer('batch_size', 256, 'Batch size')
tf.app.flags.DEFINE_integer('print_every', 200, 'Print after every so training steps')
tf.app.flags.DEFINE_integer('decay_lr_steps', 3000, 'Learning rate decay interval')
# Setting ext_time to 2 results in double sequence length 1568
tf.app.flags.DEFINE_integer('ext_time', 1, 'Duration of every pixel in time-steps')

# Training Parameters
learning_rate = tf.Variable(FLAGS.lr, dtype=tf.float32, trainable=False)
decay_learning_rate_op = tf.assign(learning_rate, learning_rate * 0.95)
training_steps = FLAGS.training_steps
batch_size = FLAGS.batch_size
display_step = FLAGS.print_every

# Network Parameters
num_input = 1  # MNIST data input (img shape: 28*28)
timesteps = 28 * 28  # timesteps
num_hidden = FLAGS.num_hidden  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps*FLAGS.ext_time, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
w_out = tf.Variable(tf.random_normal([num_hidden, num_classes]))
b_out = tf.Variable(tf.random_normal([num_classes]))


def RNN(x, readout_weight, readout_biase):
    x = tf.unstack(x, timesteps*FLAGS.ext_time, 1)

    if FLAGS.model == 'LSTM':
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    elif FLAGS.model == 'RNN':
        lstm_cell = rnn.BasicRNNCell(num_hidden)
    else:
        raise NotImplementedError("Unknown model: " + FLAGS.model)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], readout_weight) + readout_biase


logits = RNN(X, w_out, b_out)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Save parameters and training log
try:
    flag_dict = FLAGS.flag_values_dict()
except:
    print('Deprecation WARNING: with tensorflow >= 1.5 we should use FLAGS.flag_values_dict() to transform to dict')
    flag_dict = FLAGS.__flags
print(json.dumps(flag_dict, indent=4))

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("TOTAL PARAMS", total_parameters)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        if FLAGS.ext_time > 1:
            batch_x = np.repeat(batch_x, FLAGS.ext_time, axis=1)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps*FLAGS.ext_time, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % FLAGS.decay_lr_steps == 0:
            old_lr = sess.run(learning_rate)
            new_lr = sess.run(decay_learning_rate_op)
            print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr,new_lr))

        if step % display_step == 0 or step == 1:
            batch_x, batch_y = mnist.test.next_batch(batch_size * 4)
            if FLAGS.ext_time > 1:
                batch_x = np.repeat(batch_x, FLAGS.ext_time, axis=1)
            batch_x = batch_x.reshape((batch_size * 4, timesteps*FLAGS.ext_time, num_input))

            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Test Accuracy= " + \
                  "{:.3f}".format(acc))
            # NOTE: Test set accuracy should not be used for early stopping!!!
            # As we train the models with a fixed number of iterations we just
            # use the test set accuracy to monitor the progress

    print("Optimization Finished!")

    # Calculate accuracy for all mnist test images
    test_len = 256
    n_test_batches = (mnist.test.num_examples//test_len) + 1
    test_accuracy = []
    for i in range(n_test_batches):  # cover the whole test set
        test_data, test_label = mnist.test.next_batch(batch_size, shuffle=False)
        test_data = test_data.reshape((-1, timesteps, num_input))
        if FLAGS.ext_time > 1:
            test_data = np.repeat(test_data, FLAGS.ext_time, axis=1)

        test_accuracy.append(sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    print('''Statistics on the test set average accuracy {:.4g} +- {:.4g} (averaged over {} test batches of size {})'''
          .format(np.mean(test_accuracy), np.std(test_accuracy), n_test_batches, test_len))
