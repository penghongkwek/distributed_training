from tornado import web

from webserver.handlers import BaseHandler

class APIHandler(BaseHandler):

   def get(self):
        self.write('.....')
        self.finish()

default_handlers = [
    (r'/apibase', APIHandler),
 ]