import dynet as dy
import math


class Linear:
    "Construct a layernorm module (See citation for details)."

    def __init__(
        self,
        model: dy.ParameterCollection,
        in_dim: int,
        out_dim: int,
        init: dy.PyInitializer = None,
        bias: bool = True):

        pc = model.add_subcollection()
        if not init: init = dy.UniformInitializer(math.sqrt(in_dim))
        self.W = pc.add_parameters((out_dim, in_dim), init=init)
        if bias: self.b = pc.add_parameters((out_dim,), init=init)
        self.pc = pc
        self.bias = bias

    def __call__(self, x):
        return self.W * x + (self.b if self.bias else None)

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        in_dim, out_dim, init, bias = spec
        return Linear(model, in_dim, out_dim, init, bias)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc