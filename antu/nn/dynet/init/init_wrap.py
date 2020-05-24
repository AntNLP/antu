from typing import Tuple
import dynet as dy
from . import OrthogonalInitializer


def init_wrap(
        init: dy.PyInitializer,
        size: Tuple[int]) -> dy.PyInitializer:

    if init == OrthogonalInitializer:
        return dy.NumpyInitializer(init.init(size))
    elif isinstance(init, dy.PyInitializer) == True:
        return init
    else:
        raise RuntimeError('%s is not a instance of dy.PyInitializer.' % init)
        
