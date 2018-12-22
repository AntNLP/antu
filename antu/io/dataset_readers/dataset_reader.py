from typing import Dict, List
from abc import ABCMeta, abstractmethod
from antu.io.instance import Instance


class DatasetReader(metaclass=ABCMeta):

    @abstractmethod
    def read(self, file_path: str) -> List[Instance]:
        pass

    @abstractmethod
    def input_to_instance(self, inputs: str) -> Instance:
        pass