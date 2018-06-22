import time
import tornado.ioloop
import tornado.httpclient
import tensorflow as tf

from tornado import web
from tornado.ioloop import IOLoop
from tornado.platform.asyncio import AsyncIOMainLoop

from webserver.modelserver.tensorflow_model_server import TensorFlowModelServer

from traitlets import (
    Dict, List, Int, Float, Unicode, Bool, Integer
)
from traitlets.config import Application, catch_config_error

# from webserver.handlers import default_handlers
# from webserver.apihandlers import default_handlers as api_handlers
from webserver import apihandlers, handlers

class NoStart(Exception):
    """Exception to raise when an application shouldn't start"""

class WebServer(Application):


    port = Integer(8081,
        help="""The internal port for the Hub process.
        
        This is the internal port of the hub itself. It should never be accessed directly.
        See JupyterHub.port for the public port to use when accessing jupyterhub.
        It is rare that this port should be set except in cases of port conflict.
        """
    ).tag(config=True)
    ip = Unicode('127.0.0.1',
        help="""The ip address for the Hub process to *bind* to.
        
        By default, the hub listens on localhost only. This address must be accessible from 
        the proxy and user servers. You may need to set this to a public ip or '' for all 
        interfaces if the proxy or user servers are in containers or on a different host.

        See `hub_connect_ip` for cases where the bind and connect address should differ.
        """
    ).tag(config=True)


    config_file = Unicode('config.py',
        help="The config file to load",
    ).tag(config=True)

    tornado_settings = Dict(
        help="Extra settings overrides to pass to the tornado application."
    ).tag(config=True)

    handlers = List()

    async def start(self):
        """Start the whole thing"""
        self.io_loop = loop = IOLoop.current()

        if self.subapp:
            self.subapp.start()
            loop.stop()
            return


    def init_handlers(self):
        h = []
        # set default handlers
        h.extend(handlers.default_handlers)
        h.extend(apihandlers.default_handlers)
        self.handlers.extend(h)

    def init_tornado_application(self):
        properties = None
        """Instantiate the tornado Application object"""
        self.tornado_application = web.Application(self.handlers, **self.tornado_settings)

        print("\nWeb Server listening from port ", self.config.WebServer.port)
        self.tornado_application.listen(self.config.WebServer.port)

    @catch_config_error
    async def initialize(self, *args):
        self.load_config_file(self.config_file)
        self.init_handlers()
        self.init_tornado_application()

    async def launch_instance_async(self, argv=None):
        try:
            await self.initialize(argv)
            await self.start()
        except Exception as e:
            self.log.exception("%s", e)
            self.exit(1)


    @classmethod
    def launch_instance(cls, argv=None, **kwargs):
        """Launch an instance of a WebServer Application"""
        self = cls.instance()
        AsyncIOMainLoop().install()
        loop = IOLoop.current()
        loop.add_callback(self.launch_instance_async, argv)
        try:
            loop.start()
        except KeyboardInterrupt:
            print("\nKeyboard Interrupted")


main = WebServer.launch_instance


if __name__ == "__main__":
    main()
	
