import numpy as np


def standardize_lattice(lattice):
    """
    Convert the lattice vector into the standard format used through this package. \
    The standard format is an n x n numpy array with n being the dimensional of the system, \
    such that len(lattice) will return the dimension. For 0D system, lattice = np.array([])
    """

    L = np.array(lattice)

    assert len(L.shape) <= 2, \
        ("The dimension of the lattice vector can't be greater than 2, " +
         "but it has a shape of " + str(L.shape))

    L = L.reshape(-1)

    dim = int(np.sqrt(len(L)))

    if len(L) == 0:
        return np.array([])
    elif dim**2 == len(L):
        return L.reshape((dim, dim))
    else:
        raise ValueError("The input lattice vector has", len(L), "elements, "
                         "make sure it can be converted to a square matrix. ")

