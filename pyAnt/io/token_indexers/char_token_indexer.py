from typing import Dict, List, Callable, TypeVal
from overrides import overrides

Indices = TypeVal("Indices", List[int], List[List[int]])

class CharTokenIndexer(TokenIndexer):

    def __init__(
        self,
        related_vocabs: List[str],
        transform: Callable[str, str]) -> None:
        self.related_vocabs = related_vocabs
        self.transform = transform or lambda x:x

    @overrides
    def count_vocab_items(
        self,
        token: str,
        counters: Dict[str, Dict[str, int]]) -> None:
        """
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
        """
        res = {}
        for vocab_name in self.related_vocabs:
            index_list = []

            for token in tokens:
                index_list.append([vocab[vocab_name][self.transform(ch)]
                                  for ch in token])
            res[vocab_name] = index_list
        return res

