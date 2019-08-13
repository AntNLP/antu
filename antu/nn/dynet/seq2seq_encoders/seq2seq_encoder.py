from abc import ABCMeta, abstractmethod
from typing import List
import dynet as dy


class Seq2SeqEncoder(metaclass=ABCMeta):
    """docstring for Seq2seqEncoder"""

    @abstractmethod
    def __call__(self, inputs: List[dy.Expression]) -> List[dy.Expression]:
        pass

