import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

from webserver.modelserver import My_Data_Handler

class My_Preprocess(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, data):
        self.do_process(data)

    def do_process(self,data):
        # Data Preparation
        # ==================================================

        # Load data
        print("Loading data...")
        data_handler = My_Data_Handler()
        x_text, y = data_handler.load_data_and_labels(data['data_loading_param'])

        # Build vocabulary
        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_index = -1 * int(data['data_loading_param']['dev_sample_percentage'] * float(len(y)))
        self.x_train, self.x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        self.y_train, self.y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        del x, y, x_shuffled, y_shuffled

        with open('/tmp/data/vocabsize','w') as output_file:
            output_file.write(str(len(vocab_processor.vocabulary_)))
            output_file.close()

        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(self.y_train), len(self.y_dev)))

    def get_result(self):
        return self.x_train, self.y_train, self.x_dev, self.y_dev
