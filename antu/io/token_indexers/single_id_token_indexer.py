from typing import Dict, List, Callable, TypeVar
from overrides import overrides
from antu.io.vocabulary import Vocabulary
from antu.io.token_indexers.token_indexer import TokenIndexer
Indices = TypeVar("Indices", List[int], List[List[int]])

class SingleIdTokenIndexer(TokenIndexer):

    def __init__(
        self,
        related_vocabs: List[str],
        transform: Callable[[str,], str]) -> None:
        self.related_vocabs = related_vocabs
        self.transform = transform or (lambda x:x)

    @overrides
    def count_vocab_items(
        self,
        token: str,
        counters: Dict[str, Dict[str, int]]) -> None:
        """
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
        """
        res = {}
        for index_name in self.related_vocabs:
            index_list = [vocab[index_name][self.transform(tok)]
                          for tok in tokens]
            res[index_name] = index_list
        return res

