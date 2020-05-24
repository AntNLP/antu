import dynet as dy


def leaky_relu(x, a):
    return dy.bmax(a*x, x)
