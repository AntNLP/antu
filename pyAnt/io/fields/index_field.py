from typing import List, Dict, Iterator
from overrides import overrides
from pyAnt.data.token_indexers import TokenIndexer
from pyAnt.data.vocabulary import Vocabulary

class IndexField(Field):

    def __init__(
        self,
        name: str,
        tokens: List[str],
        indexers: List[TokenIndexer] = None):
        self.name = name
        self.tokens = tokens
        self.indexers = indexers

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> str:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    @overrides
    def count_vocab_items(
        self,
        counters: Dict[str, Dict[str, int]]) -> None:
        """
        ``IndexField`` doesn't need index operation.
        """
        pass

    @overrides
    def index(
        self,
        vocab: Vocabulary) -> None:
        """
        ``IndexField`` doesn't need index operation.
        """
        pass



