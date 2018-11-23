from typing import Dict, List, Generic, TypeVar
import abc


class Dataset():

    vocabs: Dict[str, Vocabulary] = {}
    datasets: Dict[str, List[Instance]] = {}

    @abc.abstractmethod
    def add_dataset(name: str, path: str, reader: DatasetReader) -> None:
        pass

    @abc.abstractmethod
    def add_vocabulary(name: str, vocab: Vocabulary) -> None:
        pass


