import json
import sys

from .base import APIHandler
from _version import __version__

class VersionAPIHandler(APIHandler):

   def get(self):
       data = {
           'version' : __version__,
       }
       self.finish(json.dumps(data))

default_handlers = [
    (r'/api/version', VersionAPIHandler),
 ]