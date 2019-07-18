import dynet as dy
import numpy as np
from . import Seq2seqEncoder
from ..init import orthonormal_initializer


class DeepBiRNNState:

    def __init__(
            self,
            fs: List[dy.RNNState],
            bs: List[dy.RNNState]):

        self.fs = fs
        self.bs = bs


class DeepBiRNNBuilder(Seq2seqEncoder):
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

    def __init__(self, model, n_layers, x_dim, h_dim, LSTMBuilder, param_init=False):
        pc = model.add_subcollection()
        self.DeepBiLSTM = []
        f = LSTMBuilder(1, x_dim, h_dim, pc)
        b = LSTMBuilder(1, x_dim, h_dim, pc)
        self.DeepBiLSTM.append((f, b))
        for i in range(n_layers-1):
            f = LSTMBuilder(1, h_dim*2, h_dim, pc)
            b = LSTMBuilder(1, h_dim*2, h_dim, pc)
            self.DeepBiLSTM.append((f, b))

        self.param_init = param_init
        self.pc = pc
        self.spec = (n_layers, x_dim, h_dim,
                     LSTMBuilder, param_init)

    def initial_state(vecs=None, update=True):
        fs = []
        bs = []
        if vecs:
            for f, b, vec in zip(self.DeepBiLSTM, vecs):
                fs.append(f.initial_state(vec[0]))
                bs.append(b.initial_state(vec[1]))
        else:
            for f, b in self.DeepBiLSTM:
                fs.append(f.initial_state())
                bs.append(b.initial_state())
        return DeepBiRNNState(fs, bs)

    def initial_state_from_raw_vectors(vecs, update=True):
        fs = []
        bs = []
        for f, b, vec in zip(self.DeepBiLSTM, vecs):
            fs.append(f.initial_state(vec[0], update))
            bs.append(b.initial_state(vec[1], update))
        return DeepBiRNNState(fs, bs)


    def set_dropout_masks(batch_size=1):
        pass

    def set_dropouts(p_x, p_h):
        pass

    def __call__(self, inputs, init_vecs=None, dropout_x=0., dropout_h=0., train=False):
        batch_size = inputs[0].dim()[1]
        if not self.fb_fusion:
            if self.param_init:
                f, b = self.f.initial_state(
                    self.f_init), self.b.initial_state(self.b_init)
            elif init_vecs:
                f, b = self.f.initial_state(
                    init_vecs["fwd"]), self.b.initial_state(init_vecs["bwd"])
            else:
                f, b = self.f.initial_state(), self.b.initial_state()
            if train:
                self.f.set_dropouts(dropout_x, dropout_h)
                self.f.set_dropout_masks(batch_size)
                self.b.set_dropouts(dropout_x, dropout_h)
                self.b.set_dropout_masks(batch_size)
            else:
                self.f.set_dropouts(0., 0.)
                self.f.set_dropout_masks(batch_size)
                self.b.set_dropouts(0., 0.)
                self.b.set_dropout_masks(batch_size)
            f_in, b_in = inputs, reversed(inputs)
            f_out, b_out = f.add_inputs(f_in), b.add_inputs(b_in)
            f_last, b_last = f_out[-1].s(), b_out[-1].s()
            f_out, b_out = [state.h()[-1]
                            for state in f_out], [state.h()[-1] for state in b_out]
            out = [dy.concatenate([f, b])
                   for f, b in zip(f_out, reversed(b_out))]
            last = [dy.concatenate([f, b]) for f, b in zip(f_last, b_last)]
            return (last, out)

        else:
            for f_lstm, b_lstm in self.DeepBiLSTM:
                f, b = f_lstm.initial_state(
                    update=True), b_lstm.initial_state(update=True)
                if train:
                    f_lstm.set_dropouts(dropout_x, dropout_h)
                    f_lstm.set_dropout_masks(batch_size)
                    b_lstm.set_dropouts(dropout_x, dropout_h)
                    b_lstm.set_dropout_masks(batch_size)
                else:
                    f_lstm.set_dropouts(0., 0.)
                    f_lstm.set_dropout_masks(batch_size)
                    b_lstm.set_dropouts(0., 0.)
                    b_lstm.set_dropout_masks(batch_size)
                fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
                inputs = [dy.concatenate([f, b])
                          for f, b in zip(fs, reversed(bs))]
            return inputs

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        n_layers, x_dim, h_dim, LSTMBuilder, param_init, fb_fusion = spec
        return DeepBiLSTMBuilder(model, n_layers, x_dim, h_dim, LSTMBuilder, param_init, fb_fusion)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc


def orthonormal_VanillaLSTMBuilder(n_layers, x_dim, h_dim, pc):
    builder = dy.VanillaLSTMBuilder(n_layers, x_dim, h_dim, pc)

    for layer, params in enumerate(builder.get_parameters()):
        W = orthonormal_initializer(
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
        W = orthonormal_initializer(
            h_dim, h_dim + (h_dim if layer > 0 else x_dim))
        W_h, W_x = W[:, :h_dim], W[:, h_dim:]
        params[0].set_value(np.concatenate([W_x]*4, 0))
        params[1].set_value(np.concatenate([W_h]*4, 0))
        b = np.zeros(4*h_dim, dtype=np.float32)
        b[h_dim:2*h_dim] = -1.0
        params[2].set_value(b)
    return builder
