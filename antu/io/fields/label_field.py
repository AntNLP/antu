from typing import List, Iterator, Dict
from overrides import overrides
from antu.io.token_indexers.token_indexer import TokenIndexer
from antu.io.vocabulary import Vocabulary
from antu.io.fields.field import Field


class LabelField(Field):

    def __init__(
        self,
        name: str,
        token: str,
        indexers: List[TokenIndexer]):
        self.name = name
        self.token = token
        self.indexers = indexers

    def __iter__(self) -> Iterator[str]:
        return iter(self.token)

    def __getitem__(self, idx: int) -> str:
        return self.token[idx]

    def __len__(self) -> int:
        return len(self.token)

    @overrides
    def count_vocab_items(
        self,
        counters: Dict[str, Dict[str, int]]) -> None:
        for idxer in self.indexers:
            idxer.count_vocab_items(token, counters)

    @overrides
    def index(
        self,
        vocab: Vocabulary) -> None:
        self.indexes = {}
        for idxer in self.indexers:
            self.indexes.update(idxer.tokens_to_indices([self.token], vocab))




