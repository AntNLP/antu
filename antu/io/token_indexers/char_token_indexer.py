from typing import Dict, List, Callable, TypeVar
from overrides import overrides
from antu.io.vocabulary import Vocabulary
from antu.io.token_indexers.token_indexer import TokenIndexer
Indices = TypeVar("Indices", List[int], List[List[int]])

class CharTokenIndexer(TokenIndexer):
    """
    A ``CharTokenIndexer`` determines how string token get represented as
    arrays of list of character indices in a model.

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
        transform: Callable[[str,], str]=lambda x:x) -> None:
        self.related_vocabs = related_vocabs
        self.transform = transform

    @overrides
    def count_vocab_items(
        self,
        token: str,
        counters: Dict[str, Dict[str, int]]) -> None:
        """
        Each character in the token is counted directly as an element.

        Parameters
        ----------
        counter : ``Dict[str, Dict[str, int]]``
            We count the number of strings if the string needs to be counted to
            some counters.
        """
        for vocab_name in self.related_vocabs:
            if vocab_name in counters:
                for ch in token:
                    counters[vocab_name][self.transform(ch)] += 1

    @overrides
    def tokens_to_indices(
        self,
        tokens: List[str],
        vocab: Vocabulary) -> Dict[str, List[List[int]]]:
        """
        Takes a list of tokens and converts them to one or more sets of indices.
        During the indexing process, each token item corresponds to a list of
        index in the vocabulary.

        Parameters
        ----------
        vocab : ``Vocabulary``
            ``vocab`` is used to get the index of each item.
        """
        res = {}
        for vocab_name in self.related_vocabs:
            index_list = []

            for token in tokens:
                index_list.append(
                    [vocab.get_token_index(self.transform(ch), vocab_name)
                    for ch in token])
            res[vocab_name] = index_list
        return res

