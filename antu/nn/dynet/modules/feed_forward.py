import dynet as dy
from ..functional import GELU
from . import dy_model


@dy_model
class PositionwiseFeedForward:
    "Implements FFN equation."

    def __init__(
            self, 
            model: dy.ParameterCollection, 
            in_dim: int, 
            hid_dim: int, 
            p: int = 0.1):

        pc = model.add_subcollection()
        self.W1 = Linear(pc, in_dim, hid_dim)
        self.W2 = Linear(pc, hid_dim, in_dim)
        self.p = p
        self.pc = pc
        self.spec = (in_dim, hid_dim, p)

    def __call__(self, x, is_train=False):
        p = self.p if is_train else 0
        return self.W2(dy.dropout(GELU(self.W1(x)), p))

