from typing import Dict, List
import abc

from

class Dataset(metaclass=abc.ABCMeta):

    vocabulary_set: Dict[str, Vocabulary] = {}
    datasets: Dict[str, List[Instance]] = {}



    @abc.abstractmethod
    def read():
        pass

    @abc.abstractmethod
    def input_to_instance():
        pass


