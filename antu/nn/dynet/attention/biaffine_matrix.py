import _dynet as dy
import numpy as np
from ..modules import dy_model
from ..init import init_wrap


@dy_model
class BiaffineMatAttention(object):
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
        h_bias=False, s_bias=False, init=dy.ConstInitializer(0.)):
        pc = model.add_subcollection()
        h_dim += s_bias
        s_dim += h_bias
        init_U = init_wrap(init, (h_dim*n_label, s_dim))
        self.U = pc.add_parameters((h_dim*n_label, s_dim), init=init_U)
        self.h_dim, self.s_dim, self.n_label = h_dim, s_dim, n_label
        self.pc, self.h_bias, self.s_bias = pc, h_bias, s_bias
        self.spec = (h_dim, s_dim, n_label, h_bias, s_bias, init)

    def __call__(self, h, s):
        # hT -> ((L, h_dim), B), s -> ((s_dim, L), B)
        if len(h.dim()[0]) == 2:
            L = h.dim()[0][1]
            if self.h_bias: s = dy.concatenate([s, dy.inputTensor(np.ones((1, L), dtype=np.float32))])
            if self.s_bias: h = dy.concatenate([h, dy.inputTensor(np.ones((1, L), dtype=np.float32))])
        else:
            if self.h_bias: s = dy.concatenate([s, dy.inputTensor(np.ones((1,), dtype=np.float32))])
            if self.s_bias: h = dy.concatenate([h, dy.inputTensor(np.ones((1,), dtype=np.float32))])
        hT = dy.transpose(h)
        lin = self.U * s        # ((h_dim*n_label, L), B)
        if self.n_label > 1:
            lin = dy.reshape(lin, (self.h_dim, self.n_label))
        
        blin = hT * lin
        if self.n_label == 1:
            return blin
        else:
            return dy.transpose(blin)


    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        h_dim, s_dim, n_label, h_bias, s_bias, init = spec
        return BiaffineMatLabelClassifier(model, h_dim, s_dim, n_label, h_bias, s_bias, init)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc
