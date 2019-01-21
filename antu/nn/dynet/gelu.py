import dynet as dy
import math


class GELU:
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def __call__(self, x):
        return 0.5 * x * (1 + dy.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * dy.pow(x, 3))))