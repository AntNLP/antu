from typing import Tuple, List
import numpy, os, gzip


def glove_reader(file_path: str) -> Tuple[List[str], List[List[float]]]:
    if os.path.isfile(file_path):
        word = []
        vector = []
        if '.gz' != file_path[-3:]:
            with open(file_path, 'r') as fp:
                for w in fp:
                    w_list = w.strip().split(' ')
                    if len(w_list) <= 2: continue
                    word.append(w_list[0])
                    vector.append([float(f) for f in w_list[1:]])
            return (word, vector)
        else:
            with gzip.open(file_path, 'rt') as fp:
                for w in fp:
                    w_list = w.strip().split(' ')
                    if len(w_list) <= 2: continue
                    word.append(w_list[0])
                    vector.append([float(f) for f in w_list[1:]])
            return (word, vector)
    else: raise RuntimeError("Glove file (%s) does not exist.")
