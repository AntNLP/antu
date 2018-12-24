from typing import Dict, Optional, Union, List, Set, TypeVar
import bidict

DEFAULT_PAD_TOKEN = "*@PAD@*"
DEFAULT_UNK_TOKEN = "*@UNK@*"

class Vocabulary(object):
    """
    Parameters
    ----------
    counters : ``Dict[str, Dict[str, int]]``, optional (default=``None``)
    min_count : ``Dict[str, int]``, optional (default=``None``)
    pretrained_vocab : ``Dict[str, List[str]]``, optional (default=``None``)
    no_pad_namespace : ``Set[str]``, optional (default=``None``)
    no_unk_namespace : ``Set[str]``, optional (default=``None``)
    """

    def __init__(self,
                 counters: Dict[str, Dict[str, int]] = None,
                 min_count: Dict[str, int] = None,
                 pretrained_vocab: Dict[str, List[str]] = None,
                 intersection_vocab: Dict[str, str] = None,
                 no_pad_namespace: Set[str] = None,
                 no_unk_namespace: Set[str] = None):

        self._PAD_token = DEFAULT_PAD_TOKEN
        self._UNK_token = DEFAULT_UNK_TOKEN
        self.min_count = min_count or {}
        self.intersection_vocab = intersection_vocab or {}
        self.no_unk_namespace = no_unk_namespace or set()
        self.no_pad_namespace = no_pad_namespace or set()
        self.vocab_cnt = {}

        for vocab_name, counter in dict(counters, **pretrained_vocab):
            self.vocab[vocab_name] = bidict()
            cnt = 0

            # Handle unknown token
            if not no_unk_namespace and vocab_name not in no_unk_namespace:
                self.vocab[vocab_name][self._UNK_token] = cnt
                cnt += 1

            # Handle padding token
            if not no_pad_namespace and vocab_name not in no_pad_namespace:
                self.vocab[vocab_name][self._PAD_token] = cnt
                cnt += 1

            # Build Vocabulary from Dataset Counter
            if isinstance(counter, dict):
                minn = (min_count[vocab_name]
                        if min_count and vocab_name in min_count else 0)
                for key, value in counter.items():
                    if value >= minn:
                        self.vocab[vocab_name][key] = cnt
                        cnt += 1

            # Build Vocabulary from Pretrained Vocabulary List
            elif isinstance(counter, list):
                is_intersection = vocab_name in intersection_vocab
                target_vocab = (self.vocab[intersection_vocab[vocab_name]]
                                if is_intersection else {})
                for key in counter:
                    if not is_intersection or key in target_vocab:
                        self.vocab[vocab_name][key] = cnt
                        cnt += 1
            self.vocab_cnt[vocab_name] = cnt


    def extend_from_pretrained_vocab(
        self,
        pretrained_vocab: Dict[str, List[str]] = None,
        intersection_vocab: Dict[str, str] = None,
        no_pad_namespace: Set[str] = None,
        no_unk_namespace: Set[str] = None) -> None:
        """
        """
        self.no_unk_namespace.update(no_unk_namespace)
        self.no_pad_namespace.update(no_pad_namespace)
        self.intersection_vocab.update(intersection_vocab)
        for vocab_name, counter in pretrained_vocab:
            self.vocab[vocab_name] = bidict()

            cnt = 0
            # Handle unknown token
            if not no_unk_namespace and vocab_name not in no_unk_namespace:
                self.vocab[vocab_name][self._UNK_token] = cnt
                cnt += 1

            # Handle padding token
            if not no_pad_namespace and vocab_name not in no_pad_namespace:
                self.vocab[vocab_name][self._PAD_token] = cnt
                cnt += 1

            # Build Vocabulary from Pretrained Vocabulary List
            is_intersection = vocab_name in intersection_vocab
            target_vocab = (self.vocab[intersection_vocab[vocab_name]]
                            if is_intersection else {})
            for key in counter:
                if not is_intersection or key in target_vocab:
                    self.vocab[vocab_name][key] = cnt
                    cnt += 1
            self.vocab_cnt[vocab_name] = cnt

    def extend_from_counter(
        self,
        counters: Dict[str, Dict[str, int]],
        min_count: Union[int, Dict[str, int]] = None,
        no_pad_namespace: Set[str] = None,
        no_unk_namespace: Set[str] = None) -> None:

        self.no_unk_namespace.update(no_unk_namespace)
        self.no_pad_namespace.update(no_pad_namespace)
        self.min_count.update(min_count)

        for vocab_name, counter in counters:
            self.vocab[vocab_name] = bidict()
            cnt = 0
            # Handle unknown token
            if not no_unk_namespace and vocab_name not in no_unk_namespace:
                self.vocab[vocab_name][self._UNK_token] = cnt
                cnt += 1

            # Handle padding token
            if not no_pad_namespace and vocab_name not in no_pad_namespace:
                self.vocab[vocab_name][self._PAD_token] = cnt
                cnt += 1

            # Build Vocabulary from Dataset Counter
            minn = (min_count[vocab_name]
                    if min_count and vocab_name in min_count else 0)
            for key, value in counter.items():
                if value >= minn:
                    self.vocab[vocab_name][key] = cnt
                    cnt += 1
            self.vocab_cnt[vocab_name] = cnt

    def add_token_to_namespace(self, token: str, namespace: str) -> None:
        self.vocab[namespace][token] = self.vocab_cnt[namespace]
        self.vocab_cnt[namespace] += 1


