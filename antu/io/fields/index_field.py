from typing import List, Dict, Iterator
from overrides import overrides
from antu.io.token_indexers.token_indexer import TokenIndexer
from antu.io.vocabulary import Vocabulary
from antu.io.fields.field import Field


class IndexField(Field):
    """
    A ``IndexField`` is an integer field, and we can use it to store data ID.

    Parameters
    ----------
    name : ``str``
        Field name. This is necessary and must be unique (not the same as other
        field names).
    tokens : ``List[str]``
        Field content that contains a list of string.
    """
    def __init__(
        self,
        name: str,
        tokens: List[str]):
        self.name = name
        self.tokens = [int(x) for x in tokens]

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
        # self.indexes = dict()
        # self.indexes[self.name] = self.tokens
        self.indexes = self.tokens



