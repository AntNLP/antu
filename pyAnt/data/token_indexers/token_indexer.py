from typing import List, Dict, TypeVal
from abc import ABCmeta, abstractmethod

Indices = TypeVal("Indices", List[int], List[List[int]])

class TokenIndexer(metaclass=ABCmeta):
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
        """

    @abstractmethod
    def tokens_to_indices(
        self,
        tokens: List[Token],
        vocab: Vocabulary,
        namespaces: List[str]) -> List[Indices]:
        """
        Takes a list of tokens and converts them to one or more sets of indices.
        This could be just an ID for each token from the vocabulary.
        """
