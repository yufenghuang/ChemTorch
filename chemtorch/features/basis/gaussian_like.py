import numpy as np


def fc(Rij, Rc):
    """ cosine shape function with cutoff at Rc"""
    out = np.zeros_like(Rij)
    out[Rij < Rc] = 0.5 * (np.cos(Rij[Rij < Rc] * np.pi / Rc) + 1)
    return out


def g1(Rij, eta, Rs, Rc):
    """ 2-atom Symmetry function before summing with respect to i, \
        the output has a shape of len(Rij) * len(eta) * len(Rs)
    """
    Rij = np.reshape(Rij, (-1, 1, 1))
    eta = np.reshape(eta, (-1, 1))
    Rs = np.reshape(Rs, (-1))

    return np.exp(-eta*(Rij - Rs)**2) * fc(Rij, Rc)


def g2(Rij, Rik, Rjk, zeta, eta, l, Rc):
    """
    3-atom Symmetry function before summing with respect to j,k, \
    the output has a shape of len(theta_i) * len(zeta) * len(eta) * len(l)
    """
    assert len(Rij) == len(Rik) == len(Rij), "Rij, Rik, Rij and theta_i must have the same size"
    assert np.ndim(Rij) <= 1, "Rij should be a 0- or 1-dimensional array, " \
                              "but the shape of xin is " + str(Rij.shape)
    Rij = np.reshape(Rij, (-1, 1, 1, 1))
    Rik = np.reshape(Rik, (-1, 1, 1, 1))
    Rjk = np.reshape(Rjk, (-1, 1, 1, 1))
    cos = (Rij ** 2 + Rik ** 2 - Rjk ** 2) / (2 * Rij * Rik)
    zeta = np.reshape(zeta, (-1, 1, 1))
    eta = np.reshape(eta, (-1, 1))
    l = np.reshape(l, (-1))

    return 2 ** (1. - zeta) \
           * (1 + l * cos) ** zeta \
           * np.exp(-eta * (Rij ** 2 + Rik ** 2 + Rjk ** 2)) \
           * fc(Rij, Rc) * fc(Rik, Rc) * fc(Rjk, Rc)


