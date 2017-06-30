from __future__ import division, print_function, absolute_import

"""
This tutorial will introduce how to combine TFLearn and Tensorflow, using
TFLearn wrappers regular Tensorflow expressions.
"""

import tensorflow as tf
import tflearn

# ----------------------------
# Utils: Using TFLearn Trainer
# ----------------------------

# Loading MNIST complete dataset
import tflearn.datasets.mnist as mnist
XTrain, YTrain, XTest, YTest = mnist.load_data(one_hot=True)

# Define a dnn using Tensorflow
with tf.Graph().as_default():

    # Model variables
    X = tf.placeholder(tf.float32, shape=(None, 784))
    Y = tf.placeholder(tf.float32, shape=(None, 10))

    w1 = tf.Variable(tf.random_normal([784, 256]))
    b1 = tf.Variable(tf.random_normal([256]))
    w2 = tf.Variable(tf.random_normal([256, 256]))
    b2 = tf.Variable(tf.random_normal([256]))
    w3 = tf.Variable(tf.random_normal([256, 10]))
    b3 = tf.Variable(tf.random_normal([10]))

    # Multilayer perceptron
    def dnn(x):
        x = tf.tanh(tf.add(tf.matmul(x, w1), b1))
        x = tf.tanh(tf.add(tf.matmul(x, w2), b2))
        x = tf.add(tf.matmul(x, w3), b3)
        return x

    net = dnn(X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=net))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1)), tf.float32),
        name='acc')

    # Using TFLearn Trainer
    # Define a training op (op for backprop, only need 1 in this model)
    trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer,
                              metric=accuracy, batch_size=128)

    # Create Trainer, providing all training ops. Tensorboard logs stored
    # in /tmp/tflearn_logs/. It is possible to change verbose level for more
    # details logs about gradients, variables etc...
    trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=0)
    # Training for 10 epochs.
    trainer.fit({X: XTrain, Y: YTrain}, val_feed_dicts={X: XTest, Y: YTest},
                n_epoch=10, show_metric=True)