import pytest
import math
import dynet as dy
import numpy as np
from antu.nn.dynet.seq2seq_encoders import DeepBiRNNBuilder
from antu.nn.dynet.seq2seq_encoders import orthonormal_VanillaLSTMBuilder


class TestDeepBiRNNBuilder:

    def test_DeepBiRNNBuilder(self):
        pc = dy.ParameterCollection()

        ENC = DeepBiRNNBuilder(pc, 2, 50, 20, orthonormal_VanillaLSTMBuilder)
        x = [dy.random_normal((50,)) for _ in range(10)]
        y = ENC(x, p_x=0.33, p_h=0.33, train=True)
        assert len(y) == 10
        assert y[0].dim() == ((40, ), 1)

