import os
import yaml
import logging.config
import logging
import coloredlogs
import re
import json
import numpy as np
import operator
import time
import datetime
import io

import tensorflow as tf
from webserver.modelserver import ModelServer

from webserver.modelserver import My_Model
from webserver.modelserver import My_Preprocess
from webserver.modelserver import My_Data_Handler


def setup_logging(default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    if not os.path.exists('logs/'):
        os.makedirs('logs/')
    """
    | Logging Setup
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                coloredlogs.install()
            except Exception as e:
                print(e)
                print('Error in Logging Configuration. Using default configs')
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        print('Failed to load configuration file. Using default configs')


class TensorFlowModelServer(ModelServer):

    setup_logging()
    logger = logging.getLogger(__name__)

    def train_model(data):
        TensorFlowModelServer.logger.info('TODO: Preprocess')

        init_process = My_Preprocess()
        init_process.do_process(data)
        # x_train, y_train, x_dev, y_dev = init_process.get_result()
        #
        # TensorFlowModelServer.logger.info('TODO: Create Model')
        # with tf.Graph().as_default():
        #     session_conf = tf.ConfigProto(
        #         allow_soft_placement=True,
        #         log_device_placement=False)
        #     sess = tf.Session(config=session_conf)
        #     with sess.as_default():
        #         my_model = My_Model(x_train.shape[1],
        #                             y_train.shape[1],
        #                             data['hyperparameters'])
        #
        #         # Output directory for models and summaries
        #         timestamp = str(int(time.time()))
        #         out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        #         print("Writing to {}\n".format(out_dir))
        #
        #         # Summaries for loss and accuracy
        #         loss_summary = tf.summary.scalar("loss", my_model.loss)
        #         acc_summary = tf.summary.scalar("accuracy", my_model.accuracy)
        #
        #         # Train Summaries
        #         train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        #         train_summary_dir = os.path.join(out_dir, "summaries", "train")
        #         train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        #
        #         # Dev summaries
        #         dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        #         dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        #         dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        #
        #         # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        #         checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        #         checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        #         if not os.path.exists(checkpoint_dir):
        #             os.makedirs(checkpoint_dir)
        #
        #         saver = tf.train.Saver(tf.global_variables(), max_to_keep=data['training_param']['num_checkpoints'])
        #
        #         # Initialize all variables
        #         sess.run(tf.global_variables_initializer())
        #
        #         def train_step(x_batch, y_batch):
        #             """
        #             A single training step
        #             """
        #             feed_dict = {
        #                 my_model.input_x: x_batch,
        #                 my_model.input_y: y_batch,
        #             }
        #             for item in data['addi_feed_dict']['training']:
        #                 feed_dict[getattr(my_model, item['name'])] = item['value']
        #
        #             _, step, summaries, loss, accuracy = sess.run(
        #                 [my_model.train_op, my_model.global_step, train_summary_op, my_model.loss, my_model.accuracy],
        #                 feed_dict)
        #             time_str = datetime.datetime.now().isoformat()
        #             print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        #             train_summary_writer.add_summary(summaries, step)
        #
        #         def dev_step(x_batch, y_batch, writer=None):
        #             """
        #             Evaluates model on a dev set
        #             """
        #             feed_dict = {
        #                 my_model.input_x: x_batch,
        #                 my_model.input_y: y_batch,
        #             }
        #             for item in data['addi_feed_dict']['dev']:
        #                 feed_dict[getattr(my_model, item['name'])] = item['value']
        #
        #             step, summaries, loss, accuracy = sess.run(
        #                 [my_model.global_step, dev_summary_op, my_model.loss, my_model.accuracy],
        #                 feed_dict)
        #             time_str = datetime.datetime.now().isoformat()
        #             print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        #             if writer:
        #                 writer.add_summary(summaries, step)
        #
        #         # Generate batches
        #         data_handler = My_Data_Handler()
        #         batches = data_handler.batch_iter(
        #             list(zip(x_train, y_train)), data['training_param']['batch_size']
        #             , data['training_param']['num_epochs'])
        #
        #         # Training loop. For each batch...
        #         for batch in batches:
        #             x_batch, y_batch = zip(*batch)
        #             train_step(x_batch, y_batch)
        #             current_step = tf.train.global_step(sess, my_model.global_step)
        #             if current_step % data['training_param']['evaluate_every'] == 0:
        #                 print("\nEvaluation:")
        #                 dev_step(x_dev, y_dev, writer=dev_summary_writer)
        #                 print("")
        #             if current_step % data['training_param']['checkpoint_every'] == 0:
        #                 path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        #                 print("Saved model checkpoint to {}\n".format(path))

    def do_training(data, remote_ip):
        TensorFlowModelServer.logger.info('TODO: Do Training')
        TensorFlowModelServer.train_model(data)