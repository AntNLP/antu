from typing import Dict, List
import abc


class Dataset(metaclass=abc.ABCMeta):

    vocabs: Dict[str, Vocabulary] = {}
    datasets: Dict[str, List[Instance]] = {}

    @abc.abstractmethod
    def build_dataset(name: str, reader: DatasetReader) -> None:
        pass

    @abc.abstractmethod
    def
