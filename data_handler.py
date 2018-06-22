from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy
from scipy import ndimage

from six.moves import urllib

import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "data"

# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

class My_Data_Handler():

    @staticmethod
    # Download MNIST data
    def maybe_download(filename):
        """Download the data from Yann's website, unless it's already here."""
        if not tf.gfile.Exists(DATA_DIRECTORY):
            tf.gfile.MakeDirs(DATA_DIRECTORY)
        filepath = os.path.join(DATA_DIRECTORY, filename)
        if not tf.gfile.Exists(filepath):
            filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
            with tf.gfile.GFile(filepath) as f:
                size = f.size()
            print('Successfully downloaded', filename, size, 'bytes.')
        return filepath

    @staticmethod
    # Extract the images
    def extract_data(filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels].
        Values are rescaled from [0, 255] down to [-0.5, 0.5].
        """
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
            data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
            data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
            data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
            data = numpy.reshape(data, [num_images, -1])
        return data

    @staticmethod
    # Extract the labels
    def extract_labels(filename, num_images):
        """Extract the labels into a vector of int64 label IDs."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
            num_labels_data = len(labels)
            one_hot_encoding = numpy.zeros((num_labels_data, NUM_LABELS))
            one_hot_encoding[numpy.arange(num_labels_data), labels] = 1
            one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, NUM_LABELS])
        return one_hot_encoding

    @staticmethod
    # Augment training data
    def expend_training_data(images, labels):

        expanded_images = []
        expanded_labels = []

        j = 0  # counter
        for x, y in zip(images, labels):
            j = j + 1
            if j % 100 == 0:
                print('expanding data : %03d / %03d' % (j, numpy.size(images, 0)))

            # register original data
            expanded_images.append(x)
            expanded_labels.append(y)

            # get a value for the background
            # zero is the expected value, but median() is used to estimate background's value
            bg_value = numpy.median(x)  # this is regarded as background's value
            image = numpy.reshape(x, (-1, 28))

            for i in range(4):
                # rotate the image with random degree
                angle = numpy.random.randint(-15, 15, 1)
                new_img = ndimage.rotate(image, angle, reshape=False, cval=bg_value)

                # shift the image with random distance
                shift = numpy.random.randint(-2, 2, 2)
                new_img_ = ndimage.shift(new_img, shift, cval=bg_value)

                # register new training data
                expanded_images.append(numpy.reshape(new_img_, 784))
                expanded_labels.append(y)

        # images and labels are concatenated for random-shuffle at each epoch
        # notice that pair of image and label should not be broken
        expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
        numpy.random.shuffle(expanded_train_total_data)

        return expanded_train_total_data