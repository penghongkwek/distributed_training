from .base import *
from . import base, admin, model

default_handlers = []

for mod in (base, admin, model):
    default_handlers.extend(mod.default_handlers)
