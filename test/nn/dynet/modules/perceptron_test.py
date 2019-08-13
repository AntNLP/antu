import pytest
import dynet as dy
from antu.nn.dynet import MLP
from antu.nn.dynet.init import OrthogonalInitializer


class TestPerceptron:

    def test_perceptron(self):
        pc = dy.ParameterCollection()
        init = OrthogonalInitializer
        mlp = MLP(pc, [10, 8, 5], init=init)
        x = dy.random_normal((10,))
        y = mlp(x, True)
        assert y.dim() == ((5,), 1)

        mlp_batch = MLP(pc, [10, 8, 5], p=0.5, init=init)
        x = dy.random_normal((10,), batch_size=5)
        y = mlp_batch(x, True)
        assert y.dim() == ((5,), 5)
