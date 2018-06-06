import numpy as np
from .graph import adj_list2mat, adj_mat2list


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


def cart2frac(R_cart, lattice):
    """
    Convert cartesian coordinates to fractional coordinates
    :param R_cart: cartesian coordinates
    :param lattice: lattice vector with shape n x n for n-dimensional system
    :return: fractional coordinate, for n dimensinoal system, \
             the first n-columns are assumed to be fractional
    """
    R_frac = R_cart.copy()
    if 0 < len(lattice):
        R_frac[:, :len(lattice)] = np.linalg.solve(lattice.T, R_cart[:, :len(lattice)].T).T
        R_frac[:, :len(lattice)] = R_frac[:, :len(lattice)] - np.floor(R_frac[:, :len(lattice)])
    return R_frac


def frac2cart(R_frac, lattice):
    """
    Convert fractional coordinates to cartesian coordinates
    :param R_frac: input fractional coordinate, for n dimensinoal system, \
                   the first n-columns are assumed to be fractional
    :param lattice: lattice vector with shape n x n for n-dimensional system
    :return: R_cart: cartesian coordinates
    """
    R_cart = R_frac.copy()
    if 0 < len(lattice):
        R_cart[:, :len(lattice)] = np.dot(R_cart[:, :len(lattice)], lattice)
    return R_cart


def get_nb(Rcart, lattice, dcut):

    nAtoms = Rcart.shape[0]
    dim = Rcart.shape[-1]

    Rij = Rcart[None, :, :] - Rcart[:, None, :]

    Rfrac = cart2frac(Rij.reshape((-1, dim)), lattice)
    Rfrac[Rfrac>0.5] = Rfrac[Rfrac>0.5]-1
    Rij = frac2cart(Rfrac, lattice).reshape((nAtoms, -1, dim))

    d2ij = np.sum(Rij ** 2, axis=-1)
    Rij[d2ij > dcut ** 2] = 0

    adjMat = np.array((0 < d2ij) & (d2ij < dcut ** 2), dtype=int)

    idxNb, Rij = adj_mat2list(adjMat, Rij)

    maxNb = idxNb.shape[1]

    return idxNb, Rij, maxNb


def get_nblist(Rcart, lattice, dcut):

    nAtoms = Rcart.shape[0]
    dim = Rcart.shape[-1]

    Rij = Rcart[None, :, :] - Rcart[:, None, :]

    Rfrac = cart2frac(Rij.reshape((-1, dim)), lattice)
    Rfrac[Rfrac>0.5] = Rfrac[Rfrac>0.5]-1
    Rij = frac2cart(Rfrac, lattice).reshape((nAtoms, -1, dim))

    d2ij = np.sum(Rij ** 2, axis=-1)

    adjMat = np.array((0 < d2ij) & (d2ij < dcut ** 2), dtype=int)

    nblist = adj_mat2list(adjMat)

    return nblist, adjMat


def get_distances(Rij):
    dij = np.sqrt(np.sum(Rij**2, axis=-1))
    dijk = np.sqrt(np.sum((Rij[:, :, None, :] - Rij[:, None, :, :])**2, axis=-1))
    dijk[dij == 0] = 0
    dijk.transpose([0, 2, 1])[dij == 0] = 0
    Rhat = np.zeros_like(Rij)
    Rhat[dij > 0] = Rij[dij > 0] / dij[dij > 0, None]
    return dij, dijk, Rhat
