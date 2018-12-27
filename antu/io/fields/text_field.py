from typing import List, Iterator, Dict
from overrides import overrides
from antu.io.token_indexers.token_indexer import TokenIndexer
from antu.io.vocabulary import Vocabulary
from antu.io.fields.field import Field


class TextField(Field):
    """
    A ``TextField`` is a data field that is commonly used in NLP tasks, and we
    can use it to store text sequences such as sentences, paragraphs, POS tags,
    and so on.

    Parameters
    ----------
    name : ``str``
        Field name. This is necessary and must be unique (not the same as other
        field names).
    tokens : ``List[str]``
        Field content that contains a list of string.
    indexers : ``List[TokenIndexer]``, optional (default=``list()``)
        Indexer list that defines the vocabularies associated with the field.
    """
    def __init__(
        self,
        name: str,
        tokens: List[str],
        indexers: List[TokenIndexer] = list()):
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
        We count the number of strings if the string needs to be counted to some
         counters. You can pass directly if there is no string that needs
        to be counted.

        Parameters
        ----------
        counters : ``Dict[str, Dict[str, int]]``
            Element statistics for datasets. if field indexers indicate that
            this field is related to some counters, we use field content to
            update the counters.
        """
        for idxer in self.indexers:
            for token in self.tokens:
                idxer.count_vocab_items(token, counters)

    @overrides
    def index(
        self,
        vocab: Vocabulary) -> None:
        """
        Gets one or more index mappings for each element in the Field.

        Parameters
        ----------
        vocab : ``Vocabulary``
            ``vocab`` is used to get the index of each item.
        """
        self.indexes = {}
        for idxer in self.indexers:
            self.indexes.update(idxer.tokens_to_indices(self.tokens, vocab))