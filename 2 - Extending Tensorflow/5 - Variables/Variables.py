"""
This example introduces the use of TFLearn variables to easily implement
Tensorflow variables with custom initialization and regularization.
Note: If you are using TFLearn layers, inititalization and regularization
are directly defined at the layer definition level and applied to inner
variables.
"""

import tensorflow as tf
import tflearn
import tflearn.variables as va

# Loading MNIST dataset
import tflearn.datasets.mnist as mnist
XTrain, YTrain, XTest, YTest = mnist.load_data(one_hot=True)

# Define a dnn using Tensorflow
with tf.Graph().as_default():

    # Model variables
    X = tf.placeholder(tf.float32, shape=[None, 784])
    Y = tf.placeholder(tf.float32, shape=[None, 10])

    # Multilayer perceptron
    def dnn(x):
        with tf.variable_scope('Layer1'):
            # Creating variable using TFLearn
            w1 = va.variable(name='W', shape=[784, 256], initializer='uniform_scaling', regularizer='L2')
            b1 = va.variable(name='B', shape=[256])

            x = tf.tanh(tf.add(tf.matmul(x, w1), b1))

        with tf.variable_scope('Layer2'):
            w2 = va.variable(name='W', shape=[256, 256], initializer='uniform_scaling', regularizer='L2')
            b2 = va.variable(name='B', shape=[256])

            x = tf.tanh(tf.add(tf.matmul(x, w2), b2))

        with tf.variable_scope('Layer3'):
            w3 = va.variable(name='W', shape=[256, 10], initializer='uniform_scaling')
            b3 = va.variable(name='B', shape=[10])

            x = tf.add(tf.matmul(x, w3), b3)

        return x


    net = dnn(X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1)), tf.float32),
        name='acc')

    # Define a train op
    trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer,
                              metric=accuracy, batch_size=128)

    trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=3,
                              tensorboard_dir='/tmp/tflearn_logs/')
    # Training for 10 epochs.
    trainer.fit({X: XTrain, Y: YTrain}, val_feed_dicts={X: XTest, Y: YTest},
                n_epoch=10, show_metric=True, run_id='Variables_example')
