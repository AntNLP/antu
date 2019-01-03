from typing import Tuple, List
import numpy, os


def glove_reader(file_path: str) -> Tuple[List[str], List[List[float]]]:
    if os.path.isfile(file_path):
        word = []
        vector = []
        with open(file_path, 'r') as fp:
            for w in fp:
                w_list = w.strip().split(' ')
                word.append(w_list[0])
                vector.append([float(f) for f in w_list[1:]])
        return (word, vector)
    else: raise RuntimeError("Glove file (%s) does not exist.")
