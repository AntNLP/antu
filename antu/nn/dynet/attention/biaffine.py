import _dynet as dy
import numpy as np
from antu.nn.dynet.initializer import orthonormal_initializer


class BiaffineAttention(object):
    """This builds Pointer Networks labeled Classifier:
    .. math::
        \\begin{split}
           f_{ptr}(\\boldsymbol{h}_i, \\boldsymbol{s}_t) &=
           \\boldsymbol{V}_a{}^\\top
           \\text{tanh}(\\boldsymbol{W}_1 \\boldsymbol{h}_i +
                       \\boldsymbol{W}_2 \\boldsymbol{s}_t) \\\\
           \\boldsymbol{p}_t &= \\text{softmax}(f_{ptr}(\\boldsymbol{h}_i, \\boldsymbol{s}_t)) \\\\
        \end{split}
    :param model dynet.ParameterCollection:
    :param l_dim int: Row dimension of :math:`\\boldsymbol{V}`
    :param v_dim int: Column dimension of :math:`\\boldsymbol{V}`
    :param h_dim int: Dimension of :math:`\\boldsymbol{h}`
    :param s_dim int: Dimension of :math:`\\boldsymbol{s}`
    :returns: probatilistic vector :math:`\\boldsymbol{p}_t`
    :rtype: dynet.Expression
    """
    def __init__(
        self,
        model,
        h_dim: int, s_dim: int, n_label: int,
        bias=False, init=dy.ConstInitializer(0.)):
        pc = model.add_subcollection()
        if bias:
            if n_label == 1:
                self.B = pc.add_parameters((h_dim,), init=0)
            else:
                self.V = pc.add_parameters((n_label, h_dim+s_dim), init=0)
                self.B = pc.add_parameters((n_label,), init=0)
        if init != 'orthonormal':
            self.U = pc.add_parameters((h_dim*n_label, s_dim), init)
        else:
            self.U = pc.parameters_from_numpy(orthonormal_initializer(h_dim*n_label, s_dim))
        self.h_dim, self.s_dim, self.n_label = h_dim, s_dim, n_label
        self.pc, self.bias = pc, bias
        self.spec = (h_dim, s_dim, n_label, bias, init)

    def __call__(self, h, s):
        # hT -> ((L, h_dim), B), s -> ((s_dim, L), B)
        hT = dy.transpose(h)
        lin = self.U * s        # ((h_dim*n_label, L), B)
        if self.n_label > 1:
            lin = dy.reshape(lin, (self.h_dim, self.n_label))
        blin = hT * lin
        if self.n_label == 1:
            return blin + (hT * self.B if self.bias else 0)
        else:
            return dy.transpose(blin)+(self.V*dy.concatenate([h, s])+self.B if self.bias else 0)


    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        h_dim, s_dim, n_label, bias, init = spec
        return BiaffineLabelClassifier(model, h_dim, s_dim, n_label, bias, init)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc