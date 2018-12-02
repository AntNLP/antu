from typing import Callable, List

class ConlluReader(DatasetReader):

    def __init__(
        self,
        is_ignore_line: Callable[[List[str]], bool],
        spacer: List[str]):

        self.is_ignore_line = is_ignore_line
        self.spacer = spacer

    @overrides
    def read(self, file_path: str) -> List[Instance]:

