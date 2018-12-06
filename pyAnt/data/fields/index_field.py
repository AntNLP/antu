

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
        counters: Dict[str, Dict[str, int]],
        indexers: List[TokenIndexer]) -> None:
        """
        ``IndexField`` doesn't need index operation.
        """
        pass

    @overrides
    def index(
        self,
        vocab: Vocabulary,
        indexers: Dict[str, List[TokenIndexer]]):
        """
        ``IndexField`` doesn't need index operation.
        """
        pass



