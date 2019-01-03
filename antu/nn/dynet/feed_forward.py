import dynet as dy
from .gelu import GELU


class PositionwiseFeedForward:
    "Implements FFN equation."

    def __init__(self, model, d_model, d_ff, drop_rate=0.1):
        pc = model.add_subcollection()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.drop_rate = drop_rate
        self.activation = GELU()
        self.pc = pc

    def __call__(self, x):
        return self.w_2(dy.dropout(self.activation(self.w_1(x)), self.drop_rate))

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        d_model, d_ff, drop_rate = spec
        return PositionwiseFeedForward(model, d_model, d_ff, drop_rate)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc