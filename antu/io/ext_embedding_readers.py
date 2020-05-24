from typing import Tuple, List
import numpy
import os
import gzip


def glove_reader(file_path: str, only_word: bool = False) -> Tuple[List[str], List[List[float]]]:
    if os.path.isfile(file_path):
        word = []
        vector = []
        if '.gz' != file_path[-3:]:
            with open(file_path, 'r') as fp:
                for w in fp:
                    w_list = w.strip().split(' ')
                    word.append(w_list[0])
                    if not only_word:
                        vector.append(list(map(float, w_list[1:])))
            return word if only_word else (word, vector)
        else:
            with gzip.open(file_path, 'rt') as fp:
                for w in fp:
                    w_list = w.strip().split(' ')
                    word.append(w_list[0])
                    if not only_word:
                        vector.append(list(map(float, w_list[1:])))
            return word if only_word else (word, vector)
    else:
        raise RuntimeError("Glove file (%s) does not exist.")


def fasttext_reader(file_path: str, only_word: bool = False) -> Tuple[List[str], List[List[float]]]:
    if os.path.isfile(file_path):
        word = []
        vector = []
        if '.gz' != file_path[-3:]:
            with open(file_path, 'r') as fp:
                w_dim = int(fp.readline().strip().split(' ')[1])
                for w in fp:
                    w_list = w.strip().split(' ')
                    if len(w_list)-1 != w_dim:
                        continue
                    word.append(w_list[0])
                    if not only_word:
                        vector.append(list(map(float, w_list[1:])))
            return word if only_word else (word, vector)
        else:
            with gzip.open(file_path, 'rt') as fp:
                w_dim = int(fp.readline().strip().split(' ')[1])
                for w in fp:
                    w_list = w.strip().split(' ')
                    if len(w_list)-1 != w_dim:
                        continue
                    word.append(w_list[0])
                    if not only_word:
                        vector.append(list(map(float, w_list[1:])))
            return word if only_word else (word, vector)
    else:
        raise RuntimeError("Fasttext file (%s) does not exist.")
