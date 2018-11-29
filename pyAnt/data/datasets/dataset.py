from typing import Dict, List
from abc import ABC, abstractmethod



class Dataset(metaclass=ABC):

    vocabulary_set: Vocabulary = {}
    datasets: Dict[str, List[Instance]] = {}

    @abc.abstractmethod
    def read():
        pass

    @abc.abstractmethod
    def input_to_instance():
        pass


