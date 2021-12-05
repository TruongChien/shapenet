import pickle
from abc import ABC, abstractmethod, abstractstaticmethod


class AbstractGraphDecoder(ABC):
    def __init__(self, device='cpu'):
        self.device = device

    @staticmethod
    def decode(self, decoder_input):
        pass
