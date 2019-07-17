import dynet as dy

class LayerNorm:
    "Construct a layernorm module (See citation for details)."

    def __init__(self, model, features, eps=1e-6):
        pc = model.add_subcollection()

        self.a_2 = pc.add_parameters(features, init=1)
        self.b_2 = pc.add_parameters(features, init=0)
        self.eps = eps

    def forward(self, x):
        # mean = x.mean(-1, keepdim=True)
        mean = dy.mean_elems(x)
        # std = x.std(-1, keepdim=True)
        std = dy.std(x)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2