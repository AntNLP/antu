"""Attention mechanism module
Attention is most commonly used in many NLP tasks.
It can be used in any sequence model to look back at past states.
We have implemented some of the attention mechanisms used.
"""

import dynet as dy
from . import dy_model


@dy_model
class VanillaAttention(object):
    """This computes Additive attention:
    The original attention mechanism (Bahdanau et al., 2015)
    uses a one-hidden layer feed-forward network to calculate the attention alignment:
    .. math::
        \\begin{split}
           f_{att}(\\boldsymbol{h}_i, \\boldsymbol{s}_t) &=
           \\boldsymbol{v}_a{}^\\top
           \\text{tanh}(\\boldsymbol{W}_1 \\boldsymbol{h}_i +
                       \\boldsymbol{W}_2 \\boldsymbol{s}_t) \\\\
           \\boldsymbol{a}_t &= \\text{softmax}(f_{att}(\\boldsymbol{h}_i, \\boldsymbol{s}_t)) \\\\
           \\boldsymbol{c}_t &= \sum_i a_t^i \\boldsymbol{h}_i \\\\
        \end{split}
    :param model dynet.ParameterCollection:
    :param v_dim int: Dimension of :math:`\\boldsymbol{v}`
    :param h_dim int: Dimension of :math:`\\boldsymbol{h}`
    :param s_dim int: Dimension of :math:`\\boldsymbol{s}`
    :returns: attention vector :math:`\\boldsymbol{c}_t`
    :rtype: dynet.Expression
    """

    def __init__(self, model, v_dim, h_dim, s_dim):
        pc = model.add_subcollection()

        self.W1 = pc.add_parameters((v_dim, h_dim))
        self.W2 = pc.add_parameters((v_dim, s_dim))
        self.v = pc.add_parameters((1, v_dim))

        self.pc = pc
        self.spec = v_dim, h_dim, s_dim

    def __call__(self, s_t, h_matrix):
        e_t = self.v * dy.tanh(self.W1*h_matrix + self.W2 * s_t)
        a_t = dy.softmax(dy.transpose(e_t))
        c_t = h_matrix * a_t
        return c_t
