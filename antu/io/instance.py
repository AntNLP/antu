from typing import Dict, MutableMapping, Mapping, TypeVar, List

from antu.io.vocabulary import Vocabulary
from antu.io.fields.field import Field

Indices = TypeVar("Indices", List[int], List[List[int]])


class Instance(Mapping[str, Field]):

    def __init__(self, fields: Dict[str, Field]) -> None:
        self.fields = fields
        self.indexed = False

    def __getitem__(self, key: str) -> Field:
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def add_field(self, field_name: str, field: Field) -> None:
        """
        Add the field to the existing ``Instance``.
        """
        self.fields[field_name] = field
        if self.indexed:
            field.index(vocab)

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        Increments counts in the given ``counter`` for all of the vocabulary
        items in all of the ``Fields`` in this ``Instance``.
        """
        for field in self.fields.values():
            field.count_vocab_items(counter)

    def index_fields(self, vocab: Vocabulary) -> Dict[str, Dict[str, Indices]]:
        """
        Indexes all fields in this ``Instance`` using the provided ``Vocabulary``.
        This `mutates` the current object, it does not return a new ``Instance``.
        A ``DataIterator`` will call this on each pass through a dataset; we use the ``indexed``
        flag to make sure that indexing only happens once.
        This means that if for some reason you modify your vocabulary after you've
        indexed your instances, you might get unexpected behavior.
        """
        if not self.indexed:
            self.indexed = True
            for field in self.fields.values():
                field.index(vocab)
        res = {}
        for field_name, field in self.fields.items():
            res[field_name] = field.indexes
        return res

