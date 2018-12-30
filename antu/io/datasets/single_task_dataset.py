from typing import Dict, List
from abc import ABCMeta, abstractmethod
from antu.io.vocabulary import Vocabulary
from antu.io.instance import Instance


class SingleTaskDataset(metaclass=ABCMeta):

    vocabulary_set: Vocabulary = {}
    datasets: Dict[str, List[Instance]] = {}
    def __init__(self, vocab, datasets):


    @abstractmethod
    def read():
        pass

    @abstractmethod
    def input_to_instance():
        pass

