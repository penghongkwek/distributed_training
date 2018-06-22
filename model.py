import tensorflow as tf
import numpy as np


class My_Model(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, is_training=True):


        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, 784])
        self.input_y = tf.placeholder(tf.float32, [None, 10]) #answer
        self.is_training = tf.placeholder(tf.bool, name='MODE')

        with tf.name_scope("output"):
            batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):
                # For slim.conv2d, default argument values are like
                # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
                # padding='SAME', activation_fn=nn.relu,
                # weights_initializer = initializers.xavier_initializer(),
                # biases_initializer = init_ops.zeros_initializer,
                net = slim.conv2d(input_x, 32, [5, 5], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.conv2d(net, 64, [5, 5], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.flatten(net, scope='flatten3')

                # For slim.fully_connected, default argument values are like
                # activation_fn = nn.relu,
                # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
                # weights_initializer = initializers.xavier_initializer(),
                # biases_initializer = init_ops.zeros_initializer,
                net = slim.fully_connected(net, 1024, scope='fc3')
                net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
                self.outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')

        # Get loss of model
        with tf.name_scope("loss"):
            self.loss = slim.losses.softmax_cross_entropy(self.outputs, self.input_y)

        # Define optimizer
        with tf.name_scope("adam"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(
                1e-4,  # Base learning rate.
                self.global_step * batch_size,  # Current index into the dataset.
                train_size,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)