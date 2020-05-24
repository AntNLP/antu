import dynet as dy
import ..modules import dy_model


@dy_model
class Char2WordCNNEmbedder(object):
    """This builds char to word embedder with CNN:
    :param model dynet.ParameterCollection:
    :param n_char int: Number of char
    :param char_dim int: Dimension of char embedding
    :param n_filter int: Number of CNN filter
    :param win_sizes list: Filter width list
    :returns: c2w_emb
    :rtype: list
    """

    def __init__(self, model, n_char, char_dim, n_filter, win_sizes):
        pc = model.add_subcollection()

        self.clookup = pc.add_lookup_parameters((n_char, char_dim))
        self.Ws = [pc.add_parameters((char_dim, size, 1, n_filter),
                                     init=dy.GlorotInitializer(gain=0.5))
                   for size in win_sizes]
        self.bs = [pc.add_parameters((n_filter),
                                     init=dy.ConstInitializer(0))
                   for _ in win_sizes]

        self.win_sizes = win_sizes
        self.pc = pc
        self.spec = (n_char, char_dim, n_filter, win_sizes)

    def __call__(self, sentence, c2i, maxn_char, act, train=False):
        words_batch = []
        for token in sentence:
            chars_emb = [self.clookup[int(c2i.get(c, 0))] for c in token.chars]
            c2w = dy.concatenate_cols(chars_emb)
            c2w = dy.reshape(c2w, tuple(list(c2w.dim()[0]) + [1]))
            words_batch.append(c2w)

        words_batch = dy.concatenate_to_batch(words_batch)
        convds = [dy.conv2d(words_batch, W, stride=(
            1, 1), is_valid=True) for W in self.Ws]
        actds = [act(convd) for convd in convds]
        poolds = [dy.maxpooling2d(actd, ksize=(1, maxn_char-win_size+1), stride=(1, 1))
                  for win_size, actd in zip(self.win_sizes, actds)]
        words_batch = [dy.reshape(poold, (poold.dim()[0][2],))
                       for poold in poolds]
        words_batch = dy.concatenate([out for out in words_batch])

        c2w_emb = []
        for idx, token in enumerate(sentence):
            c2w_emb.append(dy.pick_batch_elem(words_batch, idx))
        return c2w_emb
