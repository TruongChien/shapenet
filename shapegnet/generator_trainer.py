import pickle
from abc import ABC, abstractmethod, ABCMeta
from .utils import fmt_print


class GeneratorTrainer(ABC, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, verbose=False):
        self.verbose = verbose

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @staticmethod
    def save_graphs(g, file_name, verbose=False):

        if file_name is None:
            raise Exception("File name is none.")

        if len(file_name) == 0:
            raise Exception("empty file name")

        if verbose:
            fmt_print("Saving graph to a file ", file_name)

        with open(file_name, "wb") as f:
            pickle.dump(g, f)
