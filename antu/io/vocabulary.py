from typing import Dict, Optional, Union, List, Set, TypeVar
from bidict import bidict

DEFAULT_PAD_TOKEN = "*@PAD@*"
DEFAULT_UNK_TOKEN = "*@UNK@*"


class Vocabulary(object):
    """
    Parameters
    ----------
    counters : ``Dict[str, Dict[str, int]]``, optional (default= ``dict()`` )
        Element statistics for datasets.
    min_count : ``Dict[str, int]``, optional (default= ``dict()`` )
        Defines the minimum number of occurrences when some counter are
        converted to vocabulary.
    pretrained_vocab : ``Dict[str, List[str]]``, optional (default=  ``dict()``
        External pre-trained vocabulary.
    intersection_vocab : ``Dict[str, str]``, optional (default= ``dict()`` )
        Defines the intersection with which vocabulary takes, when loading some
        oversized pre-trained vocabulary.
    no_pad_namespace : ``Set[str]``, optional (default= ``set()`` )
        Defines which vocabularies do not have `pad` token.
    no_unk_namespace : ``Set[str]``, optional (default= ``set()`` )
        Defines which vocabularies do not have `oov` token.
    """

    def __init__(self,
                 counters: Dict[str, Dict[str, int]] = dict(),
                 min_count: Dict[str, int] = dict(),
                 pretrained_vocab: Dict[str, List[str]] = dict(),
                 intersection_vocab: Dict[str, str] = dict(),
                 no_pad_namespace: Set[str] = set(),
                 no_unk_namespace: Set[str] = set()):

        self._PAD_token = DEFAULT_PAD_TOKEN
        self._UNK_token = DEFAULT_UNK_TOKEN
        self.min_count = min_count
        self.intersection_vocab = intersection_vocab
        self.no_unk_namespace = no_unk_namespace
        self.no_pad_namespace = no_pad_namespace
        self.vocab_cnt = {}
        self.vocab = {}

        for vocab_name, counter in dict(counters, **pretrained_vocab).items():
            self.vocab[vocab_name] = bidict()
            cnt = 0

            # Handle unknown token
            if vocab_name not in no_unk_namespace:
                self.vocab[vocab_name][self._UNK_token] = cnt
                cnt += 1

            # Handle padding token
            if vocab_name not in no_pad_namespace:
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
            pretrained_vocab: Dict[str, List[str]],
            intersection_vocab: Dict[str, str] = dict(),
            no_pad_namespace: Set[str] = set(),
            no_unk_namespace: Set[str] = set()) -> None:
        """
        Extend the vocabulary from the pre-trained vocabulary after defining
        the vocabulary.

        Parameters
        ----------
        pretrained_vocab : ``Dict[str, List[str]]``
            External pre-trained vocabulary.
        intersection_vocab : ``Dict[str, str]``, optional (default= ``dict()`` )
            Defines the intersection with which vocabulary takes, when loading some
            oversized pre-trained vocabulary.
        no_pad_namespace : ``Set[str]``, optional (default= ``set()`` )
            Defines which vocabularies do not have `pad` token.
        no_unk_namespace : ``Set[str]``, optional (default= ``set()`` )
            Defines which vocabularies do not have `oov` token.
        """
        self.no_unk_namespace.update(no_unk_namespace)
        self.no_pad_namespace.update(no_pad_namespace)
        self.intersection_vocab.update(intersection_vocab)
        for vocab_name, counter in pretrained_vocab.items():
            self.vocab[vocab_name] = bidict()

            cnt = 0
            # Handle unknown token
            if vocab_name not in no_unk_namespace:
                self.vocab[vocab_name][self._UNK_token] = cnt
                cnt += 1

            # Handle padding token
            if vocab_name not in no_pad_namespace:
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
            min_count: Union[int, Dict[str, int]] = dict(),
            no_pad_namespace: Set[str] = set(),
            no_unk_namespace: Set[str] = set()) -> None:
        """
        Extend the vocabulary from the dataset statistic counters after defining
        the vocabulary.

        Parameters
        ----------
        counters : ``Dict[str, Dict[str, int]]``
            Element statistics for datasets.
        min_count : ``Dict[str, int]``, optional (default= ``dict()`` )
            Defines the minimum number of occurrences when some counter are
            converted to vocabulary.
        no_pad_namespace : ``Set[str]``, optional (default= ``set()`` )
            Defines which vocabularies do not have `pad` token.
        no_unk_namespace : ``Set[str]``, optional (default= ``set()`` )
            Defines which vocabularies do not have `oov` token.
        """
        self.no_unk_namespace.update(no_unk_namespace)
        self.no_pad_namespace.update(no_pad_namespace)
        self.min_count.update(min_count)

        for vocab_name, counter in counters.items():
            self.vocab[vocab_name] = bidict()
            cnt = 0
            # Handle unknown token
            if vocab_name not in no_unk_namespace:
                self.vocab[vocab_name][self._UNK_token] = cnt
                cnt += 1

            # Handle padding token
            if vocab_name not in no_pad_namespace:
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
        """
        Extend the vocabulary by add token to vocabulary namespace.

        Parameters
        ----------
        token : ``str``
            The token that needs to be added.
        namespace : ``str``
            Which vocabulary needs to be added to.
        """
        self.vocab[namespace][token] = self.vocab_cnt[namespace]
        self.vocab_cnt[namespace] += 1

    def get_token_index(self, token: str, vocab_name: str) -> int:
        """
        Gets the index of a token in the vocabulary.

        Parameters
        ----------
        token : ``str``
            Gets the index of which token.
        namespace : ``str``
            Which vocabulary this token belongs to.

        Returns
        -------
        Index : ``int``
        """
        if token in self.vocab[vocab_name]:
            return self.vocab[vocab_name][token]
        elif vocab_name not in self.no_unk_namespace:
            return self.vocab[vocab_name][self._UNK_token]
        else:
            raise RuntimeError(
                'Try to get a OOV token (%s)\'s index from a no unknown token '
                'vocabulary (%s)' % (token, vocab_name))

    def get_token_from_index(self, index: int, vocab_name: str) -> str:
        """
        Gets the token of a index in the vocabulary.

        Parameters
        ----------
        index : ``int``
            Gets the token of which index.
        namespace : ``str``
            Which vocabulary this index belongs to.

        Returns
        -------
        Token : ``str``
        """
        if index < self.vocab_cnt[vocab_name]:
            return self.vocab[vocab_name].inv[index]
        else:
            raise RuntimeError(
                'Index (%d) out of vocabulary (%s) range'
                % (index, vocab_name))

    def get_vocab_size(self, namespace: str) -> int:
        """
        Gets the size of a vocabulary.

        Parameters
        ----------
        namespace : ``str``
            Which vocabulary.

        Returns
        -------
        Vocabulary size : ``int``
        """
        return len(self.vocab[namespace])

    def get_padding_index(self, namespace: str) -> int:
        if namespace not in self.no_pad_namespace:
            return self.vocab[namespace][self._PAD_token]
        else:
            raise RuntimeError("(%s) doesn't has PAD token." % (namespace))

    def get_unknow_index(self, namespace: str) -> int:
        if namespace not in self.no_unk_namespace:
            return self.vocab[namespace][self._UNK_token]
        else:
            raise RuntimeError("(%s) doesn't has UNK token." % (namespace))
