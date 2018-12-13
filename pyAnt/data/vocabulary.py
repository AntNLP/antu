from typing import Dict, Optional, Union
import bidict

DEFAULT_PAD_TOKEN = "*@PAD@*"
DEFAULT_UNK_TOKEN = "*@UNK@*"

class Vocabulary(object):
    """
    Parameters
    ----------
    """

    def __init__(self,
                 counters: Dict[str, Dict[str, int]] = None,
                 min_count: Dict[str, int] = None,
                 pretrained_vocab: Dict[str, List[str]] = None,
                 no_pad_namespace: Set[str] = None,
                 no_unk_namespace: Set[str] = None):
        self._pad_token = DEFAULT_PAD_TOKEN
        self._UNK_token = DEFAULT_UNK_TOKEN



    def extend_from_pretrained_files(
        self,
        pretrained_files: Dict[str, str],
        min_count: Union[int, Dict[str, int]] = None,
        intersection_vocabs: Optional[Dict[str, str]] = None) -> None:
        pass

    def extend_from_counter(
        self,
        counter: Dict[str, Dict[str, int]],
        min_count: Union[int, Dict[str, int]] = None) -> None:
        pass

    def add_token_to_namespace(self, token: str, namespace: str) -> int:
        pass
