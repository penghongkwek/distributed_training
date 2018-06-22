from abc import abstractmethod


class ModelServer(object):

    @abstractmethod
    def do_training(self):
        pass
