from typing import Dict, List
import abc

class DatasetReader(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def read(file_path):
        pass

    @abc.abstractmethod
    def input_to_instance():
        pass