import _dynet as dy
import numpy as np

class PointerLabelClassifier(object):
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
    def __init__(self, model, l_dim, v_dim, h_dim, s_dim, layers=1):
        pc = model.add_subcollection()
        self.layers = layers
        self.V  = ([pc.add_parameters((1, v_dim)) for _ in range(layers-1)]
                  +[pc.add_parameters((l_dim, v_dim))])

        self.W1 = [pc.add_parameters((v_dim, h_dim)) for _ in range(layers)]

        self.W2 = ([pc.add_parameters((v_dim, s_dim))]
                  +[pc.add_parameters((v_dim, h_dim+s_dim)) for _ in range(layers-1)])

        self.B1 = pc.add_parameters((l_dim, h_dim), init=dy.ConstInitializer(0))
        self.B2 = pc.add_parameters((l_dim, s_dim), init=dy.ConstInitializer(0))

        ## Only single layer support
        #self._W1 = pc.add_parameters((v_dim, h_dim))
        #self._W2 = pc.add_parameters((v_dim, s_dim))
        #self._V  = pc.add_parameters((l_dim, v_dim))

        self.pc = pc
        self.spec = l_dim, v_dim, h_dim, s_dim, layers

    def __call__(self, x, h_matrix, noprob=False):
        s_t = x
        for i in range(self.layers-1):
            e_t = self.V[i] * dy.tanh(self.W1[i]*h_matrix + self.W2[i]*s_t)
            a_t = dy.softmax(dy.transpose(e_t))
            c_t = h_matrix * a_t
            s_t = dy.concatenate([x, c_t])

        e_t = self.V[-1] * dy.tanh(self.W1[-1]*h_matrix + self.W2[-1]*s_t) + self.B1 * h_matrix + self.B2 * s_t

        if len(h_matrix.dim()[0]) > 1:
            e_t = dy.reshape(e_t, (self.V[-1].dim()[0][0] * h_matrix.dim()[0][1],))
        if not noprob:
            p_t = dy.softmax(e_t)
            return p_t
        else:
            return e_t

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        l_dim, v_dim, h_dim, s_dim, layers = spec
        return PointerLabelClassifier(model, l_dim, v_dim, h_dim, s_dim, layers)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc


class BiaffineLabelClassifier(object):
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
    def __init__(self, model, h_dim, s_dim, n_label, h_bias=False, s_bias=False):
        pc = model.add_subcollection()
        if h_bias: h_dim += 1
        if s_bias: s_dim += 1
        if n_label == 1:
            self.U = pc.add_parameters(
                (h_dim, s_dim), init = dy.ConstInitializer(0.))
        else:
            self.U = pc.add_parameters(
                (h_dim*n_label, s_dim), init = dy.ConstInitializer(0.))
        self.pc = pc
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.n_label = n_label
        self.h_bias = h_bias
        self.s_bias = s_bias
        self.spec = (h_dim, s_dim, n_label, h_bias, s_bias)

    def __call__(self, h, s):
        if self.h_bias:
            if len(h.dim()[0]) == 2:
                h = dy.concatenate([h, dy.inputTensor(np.ones((1, h.dim()[0][1]), dtype=np.float32))])
            else:
                h = dy.concatenate([h, dy.inputTensor(np.ones((1,), dtype=np.float32))])
        if self.s_bias:
            if len(s.dim()[0]) == 2:
                s = dy.concatenate([s, dy.inputTensor(np.ones((1, s.dim()[0][1]), dtype=np.float32))])
            else:
                s = dy.concatenate([s, dy.inputTensor(np.ones((1,), dtype=np.float32))])
        lin = self.U * s
        if self.n_label > 1:
            lin = dy.reshape(lin, (self.h_dim, self.n_label))
        blin = dy.transpose(h) * lin
        return blin

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        h_dim, s_dim, n_label, h_bias, s_bias = spec
        return BiaffineLabelClassifier(model, h_dim, s_dim, n_label, h_bias, s_bias)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc