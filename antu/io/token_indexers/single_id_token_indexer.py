from typing import Dict, List, Callable, TypeVar
from overrides import overrides
from .. import Vocabulary
from . import TokenIndexer
Indices = TypeVar("Indices", List[int], List[List[int]])


class SingleIdTokenIndexer(TokenIndexer):
    """
    A ``SingleIdTokenIndexer`` determines how string token get represented as
    arrays of single id indices in a model.

    Parameters
    ----------
    related_vocabs : ``List[str]``
        Which vocabularies are related to the indexer.
    transform : ``Callable[[str,], str]``, optional (default=``lambda x:x``)
        What changes need to be made to the token when counting or indexing.
        Commonly used are lowercase transformation functions.
    """

    def __init__(
            self,
            related_vocabs: List[str],
            transform: Callable[[str, ], str] = lambda x: x) -> None:
        self.related_vocabs = related_vocabs
        self.transform = transform

    @overrides
    def count_vocab_items(
            self,
            token: str,
            counters: Dict[str, Dict[str, int]]) -> None:
        """
        The token is counted directly as an element.

        Parameters
        ----------
        counter : ``Dict[str, Dict[str, int]]``
            We count the number of strings if the string needs to be counted to
            some counters.
        """
        for vocab_name in self.related_vocabs:
            if vocab_name in counters:
                counters[vocab_name][self.transform(token)] += 1

    @overrides
    def tokens_to_indices(
            self,
            tokens: List[str],
            vocab: Vocabulary) -> Dict[str, List[int]]:
        """
        Takes a list of tokens and converts them to one or more sets of indices.
        During the indexing process, each item corresponds to an index in the
        vocabulary.

        Parameters
        ----------
        vocab : ``Vocabulary``
            ``vocab`` is used to get the index of each item.

        Returns
        -------
        res : ``Dict[str, List[int]]``
            if the token and index list is [w1:5, w2:3, w3:0], the result will
            be {'vocab_name' : [5, 3, 0]}
        """
        res = {}
        for index_name in self.related_vocabs:
            index_list = [vocab.get_token_index(self.transform(tok), index_name)
                          for tok in tokens]
            res[index_name] = index_list
        return res
