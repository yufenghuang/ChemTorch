import numpy as np


def piecewise_cosine(Rij, num_basis, start=-1, stop=1):
    """ Piecewise cosine functions on an array xin, \
        return a matrix of dimension dim(xin)*num_basis"""

    assert np.ndim(Rij) <= 1, "Rij should be a 0- or 1-dimensional array, " \
                              "but the shape of xin is " + str(Rij.shape)
    assert num_basis >= 2, "num_basis has to be greater than 2"
    num_basis = int(num_basis)

    h = (stop - start)/(num_basis-1)
    nodes = np.linspace(start, stop, num_basis)

    y = Rij[:, None] - nodes
    out = np.zeros_like(y)

    out[np.abs(y) < h] = np.cos(y[np.abs(y) < h] * np.pi / h)/2 + 1/2

    return out


def diff_p_cosine(Rij, num_basis, start=-1, stop=1):
    """ Derivative of the piecewise cosine functions on an array xin, \
        return a matrix of dimension dim(xin)*num_basis"""

    assert np.ndim(Rij) <= 1, "Rij should be a 0- or 1-dimensional array, " \
                              "but the shape of xin is " + str(Rij.shape)
    assert num_basis >= 2, "num_basis has to be greater than 2"
    num_basis = int(num_basis)

    h = (stop - start)/(num_basis-1)
    nodes = np.linspace(start, stop, num_basis)

    y = Rij[:, None] - nodes
    out = np.zeros_like(y)

    out[np.abs(y) < h] = -1/2 * np.pi / h * np.sin(y[np.abs(y) < h] * np.pi / h)

    return out
