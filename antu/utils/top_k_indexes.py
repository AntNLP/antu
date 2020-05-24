import numpy as np


def top_k_2D_col_indexes(arr: np.array, k: int):
    assert (len(arr.shape) == 2 and k >= 0 and k <= arr.size)
    tot_size = arr.size
    num_row = arr.shape[0]
    res = np.argpartition(arr.T.reshape((tot_size,)), -k)[-k:] // num_row
    return res 
