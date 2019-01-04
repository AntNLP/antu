from abc import ABCMeta, abstractmethod
import dynet as dy

class Seq2seqEncoder(metaclass=ABCMeta):
    """docstring for Seq2seqEncoder"""

    @abstractmethod
    def __call__(self, inputs: List[dy.Expression]) -> :

