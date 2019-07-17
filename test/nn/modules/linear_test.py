import pytest
import math
import dynet as dy
import numpy as np
from antu.nn.dynet import Linear


class TestLinear:

    def test_linear(self):
        pc = dy.ParameterCollection()
        affine = Linear(pc, in_dim=10, out_dim=5, bias=True)
        x = dy.random_normal((10,))
        y = affine(x)
        assert y.dim() == ((5,), 1)

        init = dy.ConstInitializer(1)
        affine = Linear(pc, in_dim=10, out_dim=1, bias=False, init=init)
        x = dy.random_normal((10,))
        y = affine(x)
        assert math.fabs(np.sum(y.npvalue())-np.sum(x.npvalue())) < 1e-6
