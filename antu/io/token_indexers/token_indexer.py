from typing import List, Dict, TypeVar, Callable
from abc import ABCMeta, abstractmethod

from antu.io.vocabulary import Vocabulary

Indices = TypeVar("Indices", List[int], List[List[int]])

class TokenIndexer(metaclass=ABCMeta):
    """
    A ``TokenIndexer`` determines how string tokens get represented as arrays of
    indices in a model.
    """

    @abstractmethod
    def count_vocab_items(
        self,
        token: str,
        counter: Dict[str, Dict[str, int]]) -> None:
        """
        Defines how each token in the field is counted. In most cases, just use
        the string as a key. However, for character-level ``TokenIndexer``, you
        need to traverse each character in the string.

        Parameters
        ----------
        counter : ``Dict[str, Dict[str, int]]``
            We count the number of strings if the string needs to be counted to
            some counters.
        """
        pass

    @abstractmethod
    def tokens_to_indices(
        self,
        tokens: List[str],
        vocab: Vocabulary) -> Dict[str, Indices]:
        """
        Takes a list of tokens and converts them to one or more sets of indices.
        This could be just an ID for each token from the vocabulary.

        Parameters
        ----------
        vocab : ``Vocabulary``
            ``vocab`` is used to get the index of each item.
        """
        pass
