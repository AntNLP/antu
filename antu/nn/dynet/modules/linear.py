import dynet as dy
import math
from . import dy_model
from ..init import init_wrap


@dy_model
class Linear:
    "Construct a Affine Transformation."

    def __init__(
            self,
            model: dy.ParameterCollection,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            init: dy.PyInitializer = dy.GlorotInitializer()):

        pc = model.add_subcollection()
        init = init_wrap(init, (out_dim, in_dim))
        self.W = pc.add_parameters((out_dim, in_dim), init=init)
        if bias:
            self.b = pc.add_parameters((out_dim,), init=0)
        self.pc = pc
        self.bias = bias
        self.spec = (in_dim, out_dim, bias, init)

    def __call__(self, x):
        b = self.b if self.bias else 0
        return self.W * x + b 
