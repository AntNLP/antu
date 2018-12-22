import _dynet as dy

class MLP(object):
    """docstring for MLP"""
    def __init__(self, model, hidden_sizes, act=dy.tanh, bias=True, dropout=0.0):
        pc = model.add_subcollection()
        sizes = hidden_sizes
        self.W = [
            pc.add_parameters((x, y))
            for x, y in zip(sizes[1:], sizes[:-1])]
        self.b = [
            pc.add_parameters((y,), init=dy.ConstInitializer(0))
            for y in sizes[1:]]

        self.pc = pc
        self.act = act
        self.bias = bias
        self.dropout = dropout
        self.spec = (hidden_sizes, act, bias, dropout)

    def __call__(self, x, train=False):
        h = x
        for W, b in zip(self.W, self.b):
            h = self.act(dy.affine_transform([b, W, h]))
            if train:
                h = dy.dropout_dim(h, 1, self.dropout)
        return h

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        hidden_sizes, act_func, bias, dropout = spec
        return MLP(model, hidden_sizes, act_func, bias, dropout)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc