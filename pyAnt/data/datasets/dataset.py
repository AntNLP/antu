from typing import Dict, List
import abc


class Dataset(metaclass=abc.ABCMeta):

    vocabulary_set: Vocabulary = {}
    datasets: Dict[str, List[Instance]] = {}

    @abc.abstractmethod
    def read():
        pass

    @abc.abstractmethod
    def input_to_instance():
        pass


