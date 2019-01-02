from typing import Dict, List, Callable
from overrides import overrides
from antu.io.vocabulary import Vocabulary
from antu.io.instance import Instance
from antu.io.datasets.dataset import Dataset
from antu.io.dataset_readers.dataset_reader import DatasetReader
import random
from itertools import cycle


class DatasetSetting:

    def __init__(self, file_path: str, is_train: bool):
        self.file_path = file_path
        self.is_train = is_train


class SingleTaskDataset(Dataset):

    def __init__(
        self,
        vocabulary: Vocabulary,
        datasets_settings: Dict[str, DatasetSetting],
        reader: DatasetReader):
        self.vocabulary = vocabulary
        self.datasets_settings = datasets_settings
        self.datasets = dict()
        self.reader = reader
        self.ordered = False

    def build_dataset(
        self,
        counters: Dict[str, Dict[str, int]],
        min_count: Union[int, Dict[str, int]] = dict(),
        no_pad_namespace: Set[str] = set(),
        no_unk_namespace: Set[str] = set()) -> None:

        for name, setting in self.datasets_settings.items():
            self.datasets[name] = self.reader.read(setting.file_path)
            if setting.is_train:
                self.datasets[name].count_vocab_items(counters)
        self.vocabulary.extend_from_counter(
            counters, min_count, no_pad_namespace, no_unk_namespace)
        for name in self.datasets:
            self.datasets[name].index_fields(self.vocabulary)

    def get_dataset(self, name: str) -> List[Instance]:
        return self.datasets[name]

    def get_batches(self,
        name: str,
        size: int,
        ordered: bool,
        cmp: Callable[[Instance, Instance], int]):
        if ordered != self.ordered:
            if ordered: self.datasets[name].sort(cmp)
            else: random.shuffle(self.datasets[name])
            self.ordered = ordered
        num = len(self.datasets[name]) # Number of Instances
        ret = []
        for beg in range(0, num, size):
            ins_batch = self.datasets[name][beg: beg+size]
            idx_batch = [ins.index_fields(self.vocabulary) for ins in ins_batch]
            ret.append(idx_batch)
        return cycle(ret)