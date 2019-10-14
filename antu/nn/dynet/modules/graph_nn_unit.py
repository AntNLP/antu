import dynet as dy
from . import dy_model
from ..init import init_wrap


@dy_model
class GraphNNUnit(object):
    """docstring for GraphNNUnit"""

    def __init__(
            self,
            model: dy.ParameterCollection,
            h_dim: int,
            d_dim: int,
            f=dy.tanh,
            init: dy.PyInitializer = dy.GlorotInitializer()):

        pc = model.add_subcollection()
        init_W = init_wrap(init, (h_dim, d_dim))
        self.W = pc.add_parameters((h_dim, d_dim), init=init_W)
        init_B = init_wrap(init, (h_dim, h_dim))
        self.B = pc.add_parameters((h_dim, h_dim), init=init_B)

        self.pc, self.f = pc, f
        self.spec = (h_dim, d_dim, f, init)

    def __call__(self, H, D):
        return self.f(self.W * H + self.B * D)
