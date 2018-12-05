from typing import Callable, List
import re


class ConlluReader(DatasetReader):

    def __init__(
        self,
        field_list: Dict[str, Field],
        root: str,
        is_ignore_line: Callable[[List[str]], bool],
        spacer: List[str]):

        self.field_list = field_list
        self.root = root
        self.is_ignore_line = is_ignore_line
        self.spacer = spacer

    def _read(self, file_path: str) -> Instance:
        with open(file_path, 'rt') as fp:
            root_token = re.split(self.spacer, self.root)
            tokens = [[item,] for item in root_token]
            for line in fp:
                token = re.split(self.spacer, line.strip())
                if line.strip() == '':
                    if len(tokens[0]) > 1: yield tokens
                    tokens = [[item,] for item in root_token]
                elif not self.is_ignore_line(line.strip()):
                    for idx, item in enumerate(token):
                        tokens[idx].append(item)

    @overrides
    def read(self, file_path: str) -> List[Instance]:
        res = []
        for sentence in self._read(file_path):
            res.append(self.input_to_instance(sentence))
        return res

    @overrides
    def input_to_instance(self, inputs: List[List[str]]) -> Instance:
        fields = []
        for idx, name, Type in enumerate(self.field_list):
            fields.append(Type(name, inputs[idx]))
        return Instance(fields)



