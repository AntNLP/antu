from .layer_norm import LayerNorm
from . import dy_model


@dy_model
class SublayerConnection:
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, model, size, p):
        pc = model.add_subcollection()

        self.norm = LayerNorm(pc, size)
        self.p = p
        self.spec = (size, p)

    def __call__(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + dy.dropout(sublayer(self.norm(x)), self.p)
