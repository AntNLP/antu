

class Vocabulary(object):
    """
    Parameters
    ----------
    counter:
    """

    def __init__(self,
                 counter: Dict[str, Dict[str, int]] = None,
                 pretrained_files: Optional[Dict[str, str]] = None,
                 min_count: Union[int, Dict[str, int]] = None,):
        pass


    def extend_from_pretrained_files(
        self,
        pretrained_files: Dict[str, str],
        min_count: Union[int, Dict[str, int]] = None,
        intersection_vocabs: Optional[Dict[str, str]] = None): -> None
        pass

    def extend_from_counter(
        self,
        counter: Dict[str, Dict[str, int]],
        min_count: Union[int, Dict[str, int]] = None): -> None
        pass

    def add_token_to_namespace(self, token: str, namespace: str): -> int
        pass

    def