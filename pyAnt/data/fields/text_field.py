

class TextField(Field):

    def __init__(self, name: str, tokens: List[str]):
        self.name = name
        self.tokens = tokens

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> str:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    @overrides
    def count_vocab_items(
        self,
        counter: Dict[str, Dict[str, int]],
        )

