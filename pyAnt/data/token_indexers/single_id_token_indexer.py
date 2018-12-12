from typing import Dict, List


class SingleIdTokenIndexer(TokenIndexer):

    def __init__(
        self,
        related_vocabs: List[str],
        use_lowercase: bool = False) -> None:
        self.related_vocabs = related_vocabs

    @overrides
    def count_vocab_items(
        self,
        token: str,
        counters: Dict[str, Dict[str, int]]) -> None:
        """
        """
        for vocab_name in self.related_vocabs:
            counters[vocab_name][token] += 1

    @overrides
    def tokens_to_indices(
        self,
        tokens: List[str],
        field_name: str,
        vocab: Vocabulary) -> Dict[str, List[int]]:
        """
        """
        res = {}
        for name in self.related_vocabs:
            index_name = field_name + "@" + name
            index_list = [vocab[name][tok] for tok in tokens]
            res[index_name] = index_list
        return res

