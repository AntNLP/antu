import dynet as dy
from antu.nn.dynet.initializer import orthonormal_initializer


class GraphNNUnit(object):
    """docstring for GraphNNUnit"""
    def __init__(
        self,
        model,
        h_dim, s_dim, act=dy.tanh, init=0, dropout=0.0):
        pc = model.add_subcollection()
        if init != 'orthonormal':
            self.W = pc.add_parameters((h_dim, s_dim), init)
            self.B = pc.add_parameters((h_dim, h_dim), init)
        else:
            self.W = pc.parameters_from_numpy(orthonormal_initializer(h_dim, s_dim))
            self.B = pc.parameters_from_numpy(orthonormal_initializer(h_dim, s_dim))

        self.pc, self.act, self.dropout = pc, act, dropout
        self.spec = (h_dim, s_dim, act, init, dropout)

    def __call__(self, x1, x2, train=False):
        return self.act(self.W * x1 + self.B * x2)

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        h_dim, s_dim, act, init, dropout = spec
        return GraphNNUnit(model, h_dim, s_dim, act, init, dropout)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc