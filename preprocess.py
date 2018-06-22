import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

from webserver.modelserver import My_Data_Handler

VALIDATION_SIZE = 5000  # Size of the validation set.

class My_Preprocess(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self):
        None

    def do_process(self,data):
        # Data Preparation
        # ==================================================

        # Load data
        print("Loading data...")
        data_handler = My_Data_Handler()
        # Get the data.
        train_data_filename = data_handler.maybe_download('train-images-idx3-ubyte.gz')
        train_labels_filename = data_handler.maybe_download('train-labels-idx1-ubyte.gz')
        test_data_filename = data_handler.maybe_download('t10k-images-idx3-ubyte.gz')
        test_labels_filename = data_handler.maybe_download('t10k-labels-idx1-ubyte.gz')

        # Extract it into numpy arrays.
        train_data = data_handler.extract_data(train_data_filename, 60000)
        train_labels = data_handler.extract_labels(train_labels_filename, 60000)
        test_data = data_handler.extract_data(test_data_filename, 10000)
        test_labels = data_handler.extract_labels(test_labels_filename, 10000)

        # Generate a validation set.
        validation_data = train_data[:VALIDATION_SIZE, :]
        validation_labels = train_labels[:VALIDATION_SIZE, :]
        train_data = train_data[VALIDATION_SIZE:, :]
        train_labels = train_labels[VALIDATION_SIZE:, :]

    def get_result(self):
        return self.validation_data, self.validation_labels, self.x_dev, self.y_dev
