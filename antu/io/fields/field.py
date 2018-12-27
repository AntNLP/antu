from typing import List, Dict
from abc import ABCMeta, abstractmethod

from antu.io.vocabulary import Vocabulary


class Field(metaclass=ABCMeta):
    """
    A ``Field`` is an ingredient of a data instance. In most NLP tasks, ``Field``
    stores data of string types. It contains one or more indexers that map string
    data to the corresponding index. Data instances are collections of fields.
    """
    @abstractmethod
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]) -> None:
        """
        We count the number of strings if the string needs to be mapped to one
        or more integers. You can pass directly if there is no string that needs
        to be mapped.

        Parameters
        ----------
        counter : ``Dict[str, Dict[str, int]]``
        ``counter`` is used to count the number of each item. The first key
        represents the namespace of the vocabulary, and the second key represents
        the string of the item.
        """
        pass

    @abstractmethod
    def index(self, vocab: Vocabulary) -> None:
        """
        Gets one or more index mappings for each element in the Field.

        Parameters
        ----------
        vocab : ``Vocabulary``
        ``vocab`` is used to get the index of each item.
        """
        pass
