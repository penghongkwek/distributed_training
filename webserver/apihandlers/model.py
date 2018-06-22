import json
import sys
import logging
import tornado

from .base import APIHandler
from _version import __version__

from webserver.modelserver.tensorflow_model_server import TensorFlowModelServer


class TrainHandler(APIHandler):

    def post(self):
        x_real_ip = self.request.headers.get("X-Real-IP")
        remote_ip = x_real_ip or self.request.remote_ip

        json_data = tornado.escape.json_decode(self.request.body)
        TensorFlowModelServer.do_training(json_data, remote_ip)

        self.finish()

    def get(self):
        json_data = tornado.escape.json_decode(self.request.body)
        TensorFlowModelServer.do_training(json_data)
        self.finish()


default_handlers = [
    (r'/api/model/train', TrainHandler)
]

