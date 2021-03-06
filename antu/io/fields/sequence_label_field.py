from typing import List, Iterator, Dict
from overrides import overrides
from ..token_indexers import TokenIndexer
from .. import Vocabulary
from . import Field


class SequenceLabelField(Field):

    def __init__(self,
                 name: str,
                 tokens: List[str],
                 indexers: List[TokenIndexer]):
        self.name = name
        self.tokens = tokens
        self.indexers = indexers

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> str:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    def __str__(self) -> str:
        return '{}: [{}]'.format(self.name, ', '.join(self.tokens))

    @overrides
    def count_vocab_items(self, counters: Dict[str, Dict[str, int]]) -> None:
        for idxer in self.indexers:
            for token in self.tokens:
                idxer.count_vocab_items(token, counters)

    @overrides
    def index(self, vocab: Vocabulary) -> None:
        self.indexes = {}
        for idxer in self.indexers:
            self.indexes.update(idxer.tokens_to_indices(self.tokens, vocab))
