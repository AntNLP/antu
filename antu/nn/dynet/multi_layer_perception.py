import dynet as dy
from antu.nn.dynet.initializer import orthonormal_initializer


class MLP(object):
    """docstring for MLP"""
    def __init__(
        self,
        model,
        sizes, act=dy.tanh, init=0, bias=True, dropout=0.0):
        pc = model.add_subcollection()
        if init != 'orthonormal':
            self.W = [
                pc.add_parameters((x, y), init)
                for x, y in zip(sizes[1:], sizes[:-1])]
        else:
            self.W = [
                pc.parameters_from_numpy(orthonormal_initializer(x, y))
                for x, y in zip(sizes[1:], sizes[:-1])]
        if bias: self.b = [pc.add_parameters((y,), init=0) for y in sizes[1:]]

        self.pc, self.act, self.bias = pc, act, bias
        self.dropout = dropout
        self.spec = (sizes, act, init, bias, dropout)

    def __call__(self, x, train=False):
        h = x
        # for W, b in zip(self.W[:-1], self.b[:-1]):
        for i in range(len(self.W[:-1])):
            h = self.act(self.W[i]*h + (self.b[i] if self.bias else 0))
            if train: 
                if len(h.dim()[0]) > 1: h = dy.dropout_dim(h, 1, self.dropout)
                else: h = dy.dropout(h, self.dropout)
        return self.W[-1]*h + (self.b[-1] if self.bias else 0)

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        sizes, act, init, bias, dropout = spec
        return MLP(model, sizes, act, init, bias, dropout)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc