from typing import List, Iterator, Dict

from overrides import overrides

from ..token_indexers import TokenIndexer
from .. import Vocabulary
from . import Field


class LabelField(Field):

    def __init__(self,
                 name: str,
                 tokens: str,
                 indexers: List[TokenIndexer]):
        self.name = name
        self.tokens = [tokens]
        self.indexers = indexers

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> str:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens[0])

    def __str__(self) -> str:
        return '{}: {}'.format(self.name, self.tokens[0])

    @overrides
    def count_vocab_items(self, counters: Dict[str, Dict[str, int]]) -> None:
        for idxer in self.indexers:
            idxer.count_vocab_items(self.tokens[0], counters)

    @overrides
    def index(self, vocab: Vocabulary) -> None:
        self.indexes = {}
        for idxer in self.indexers:
            self.indexes.update(idxer.tokens_to_indices(self.tokens, vocab))
