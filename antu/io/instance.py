from typing import Dict, MutableMapping, Mapping, TypeVar, List

from antu.io.vocabulary import Vocabulary
from antu.io.fields.field import Field

Indices = TypeVar("Indices", List[int], List[List[int]])


class Instance(Mapping[str, Field]):
    """
    An ``Instance`` is a collection (list) of multiple data fields.

    Parameters
    ----------
    fields : ``List[Field]``, optional (default=``None``)
        A list of multiple data fields.
    """

    def __init__(self, fields: List[Field]=None) -> None:
        self.fields = fields
        self._fields_dict = {}
        for field in fields: self._fields_dict[field.name] = field
        self.indexed = False  # Indicates whether the instance has been indexed

    def __getitem__(self, key: str) -> Field:
        return self._fields_dict[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def add_field(self, field: Field) -> None:
        """
        Add the field to the existing ``Instance``.

        Parameters
        ----------
        field : ``Field``
            Which field needs to be added.
        """
        self.fields.append(field)
        if self.indexed:
            field.index(vocab)

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]) -> None:
        """
        Increments counts in the given ``counter`` for all of the vocabulary
        items in all of the ``Fields`` in this ``Instance``.

        Parameters
        ----------
        counter : ``Dict[str, Dict[str, int]]``
            We count the number of strings if the string needs to be counted to
            some counters.
        """
        for field in self.fields:
            field.count_vocab_items(counter)

    def index_fields(self, vocab: Vocabulary) -> Dict[str, Dict[str, Indices]]:
        """
        Indexes all fields in this ``Instance`` using the provided ``Vocabulary``.
        This `mutates` the current object, it does not return a new ``Instance``.
        A ``DataIterator`` will call this on each pass through a dataset; we use the ``indexed``
        flag to make sure that indexing only happens once.
        This means that if for some reason you modify your vocabulary after you've
        indexed your instances, you might get unexpected behavior.

        Parameters
        ----------
        vocab : ``Vocabulary``
            ``vocab`` is used to get the index of each item.

        Returns
        -------
        res : ``Dict[str, Dict[str, Indices]]``
            Returns the Indices corresponding to the instance. The first key is
            field name and the second key is the vocabulary name.
        """
        if not self.indexed:
            self.indexed = True
            for field in self.fields:
                field.index(vocab)
        res = {}
        for field in self.fields:
            res[field.name] = field.indexes
        return res
