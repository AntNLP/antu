from typing import List, Dict
from abc import ABCmeta, abstractmethod


class Field(metaclass=ABCmeta):
    """
    A ``Field`` is an ingredient of a data instance. In most NLP tasks, ``Field``
    stores data of string types. It contains one or more indexers that map string
    data to the corresponding index. Data instances are collections of fields.

    For example,
    """
    @abstractmethod
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        pass

    @abstractmethod
    def index(self, vocab: Vocabulary):
        pass
