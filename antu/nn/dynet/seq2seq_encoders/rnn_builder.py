import dynet as dy
import numpy as np
from . import Seq2SeqEncoder
from ..init import get_orthogonal_matrix
from ..modules import dy_model


@dy_model
class DeepBiRNNBuilder(Seq2SeqEncoder):
    """This builds deep bidirectional LSTM:

    The original attention mechanism (Bahdanau et al., 2015)
    uses a one-hidden layer feed-forward network to calculate the attention alignment:

    :param model dynet.ParameterCollection:
    :param n_layers int: Number of LSTM layers
    :param x_dim int: Dimension of LSTM input :math:`\\boldsymbol{x}`
    :param h_dim int: Dimension of LSTM hidden state :math:`\\boldsymbol{h}`
    :param LSTMBuilder dynet._RNNBuilder: Dynet LSTM type
    :param param_init bool: Initializes LSTM with parameter
    :returns: (last_output, outputs)
    :rtype: tuple
    """

    def __init__(self, model, n_layers, x_dim, h_dim, LSTMBuilder):
        pc = model.add_subcollection()
        self.DeepBiLSTM = []
        f = LSTMBuilder(1, x_dim, h_dim, pc)
        b = LSTMBuilder(1, x_dim, h_dim, pc)
        self.DeepBiLSTM.append((f, b))
        for i in range(n_layers-1):
            f = LSTMBuilder(1, h_dim*2, h_dim, pc)
            b = LSTMBuilder(1, h_dim*2, h_dim, pc)
            self.DeepBiLSTM.append((f, b))

        self.pc = pc
        self.spec = (n_layers, x_dim, h_dim, LSTMBuilder)

    def __call__(self, inputs, init_vecs=None, p_x=0., p_h=0., out_mask=None, drop_mask=False, train=False):
        batch_size = inputs[0].dim()[1]

        if out_mask is not None:
            mask = dy.inputTensor(out_mask, True)
        for fnn, bnn in self.DeepBiLSTM:
            f, b = fnn.initial_state(update=True), bnn.initial_state(update=True)
            if train:
                fnn.set_dropouts(p_x, p_h)
                bnn.set_dropouts(p_x, p_h)
                if drop_mask: 
                    fnn.set_dropout_masks(batch_size)
                    bnn.set_dropout_masks(batch_size)
            else:
                fnn.set_dropouts(0., 0.)
                bnn.set_dropouts(0., 0.)
                if drop_mask: 
                    fnn.set_dropout_masks(batch_size)
                    bnn.set_dropout_masks(batch_size)
            fs, bs = f.transduce(inputs), b.transduce(inputs[::-1])
            inputs = [dy.concatenate([f, b]) for f, b in zip(fs, bs[::-1])]
            if out_mask is not None:
                inputs = [x*mask[i] for i, x in enumerate(inputs)]
        return inputs


def orthonormal_VanillaLSTMBuilder(n_layers, x_dim, h_dim, pc):
    builder = dy.VanillaLSTMBuilder(n_layers, x_dim, h_dim, pc)

    for layer, params in enumerate(builder.get_parameters()):
        W = get_orthogonal_matrix(
            h_dim, h_dim + (h_dim if layer > 0 else x_dim))
        W_h, W_x = W[:, :h_dim], W[:, h_dim:]
        params[0].set_value(np.concatenate([W_x]*4, 0))
        params[1].set_value(np.concatenate([W_h]*4, 0))
        b = np.zeros(4*h_dim, dtype=np.float32)
        b[h_dim:2*h_dim] = -1.0
        params[2].set_value(b)
    return builder


def orthonormal_CompactVanillaLSTMBuilder(n_layers, x_dim, h_dim, pc):
    builder = dy.CompactVanillaLSTMBuilder(n_layers, x_dim, h_dim, pc)

    for layer, params in enumerate(builder.get_parameters()):
        W = get_orthogonal_matrix(
            h_dim, h_dim + (h_dim if layer > 0 else x_dim))
        W_h, W_x = W[:, :h_dim], W[:, h_dim:]
        params[0].set_value(np.concatenate([W_x]*4, 0))
        params[1].set_value(np.concatenate([W_h]*4, 0))
        b = np.zeros(4*h_dim, dtype=np.float32)
        b[h_dim:2*h_dim] = -1.0
        params[2].set_value(b)
    return builder
