from typing import Dict, List
from abc import ABCMeta, abstractmethod
from antu.io.vocabulary import Vocabulary
from antu.io.instance import Instance


class Dataset(metaclass=ABCMeta):

    vocabulary_set: Vocabulary = {}
    datasets: Dict[str, List[Instance]] = {}

    @abstractmethod
    def build_dataset():
        pass


