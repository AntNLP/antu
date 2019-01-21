from .layer_norm import LayerNorm


class SublayerConnection:
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, drop_rate):
        self.norm = LayerNorm(size)
        # .dropout = nn.Dropout(dropout)
        self.drop_rate = drop_rate

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + dy.dropout(sublayer(self.norm(x)), self.drop_rate)
