from typing import Dict, List
from abc import ABC, abstractmethod



class Dataset(metaclass=ABC):

    vocabulary_set: Vocabulary = {}
    datasets: Dict[str, List[Instance]] = {}

    @abstractmethod
    def read():
        pass

    @abstractmethod
    def input_to_instance():
        pass


