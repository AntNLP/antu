import dynet as dy

def ReLU(x):
    return dy.rectify(x)


class LeakyReLU:
    a = 0.1
    def __init__(self, a):
        LeakyReLU.a = a

    @classmethod
    def __call__(cls, x):
        return dy.bmax(LeakyReLU.a*x, x)