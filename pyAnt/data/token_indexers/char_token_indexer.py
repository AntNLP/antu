from typing import Dict, List

class CharTokenIndexer(TokenIndexer):

    def __init__(self, related_vocabs: List[str]):
        self.related_vocabs = related_vocabs

    @overrides
    def count_vocab_items(
        self,
        token: str,
        counters: Dict[str, Dict[str, int]]) -> None:
        """
        """
        for vocab_name in self.related_vocabs:
            for ch in token:
                counters[vocab_name][ch] += 1

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

