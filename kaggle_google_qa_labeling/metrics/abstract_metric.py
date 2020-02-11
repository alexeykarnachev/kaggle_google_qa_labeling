from abc import ABC, abstractmethod
import numpy as np


class AbstractMetric(ABC):
    def __init__(self):
        self.val = np.nan

    @abstractmethod
    def calculate(self, logits, labels):
        pass

    def __call__(self, logits, labels):
        self.val = self.calculate(logits=logits, labels=labels)

    def __repr__(self):
        str_ = f"{self.__class__.__name__}"
        return str_
