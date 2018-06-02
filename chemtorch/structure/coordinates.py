import numpy as np


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


def adjMat2adjList(adjMat, *values):
    # adjacency matrix to adjacency list
    # note: the indices are shifted by 1 in the adjacency lsit
    for val in values:
        assert val.shape[:2] == adjMat.shape, \
            "The first 2 dimensions of the input values must be the same as the adjacency matrix"


    idx1, idx2 = np.where(adjMat)
    maxNb = np.array([list(idx1).count(item) for item in list(idx1)]).max()
    idxNb = np.array(
        [np.concatenate([idx2[idx1 == item] + 1, np.zeros(maxNb - list(idx1).count(item), dtype=int)]) for item in
         list(set(idx1))])

    if len(values) == 0:
        return maxNb, idxNb
    else:
        outVal = [np.zeros([len(idxNb), maxNb] + list(val.shape)[2:], dtype=val.dtype) for val in values]

        for iVal, val in enumerate(values):
            (outVal[iVal])[idxNb > 0] = val[adjMat > 0]

        return tuple([maxNb, idxNb] + outVal)


def adjMat2adjList(adjMat, *values):
    # adjacency matrix to adjacency list
    # note: the indices are shifted by 1 in the adjacency lsit
    for val in values:
        assert val.shape[:2] == adjMat.shape, \
            "The first 2 dimensions of the input values must be the same as the adjacency matrix"

    idx1, idx2 = np.where(adjMat)
    maxNb = np.array([list(idx1).count(item) for item in list(idx1)]).max()
    idxNb = np.array(
        [np.concatenate([idx2[idx1 == item] + 1, np.zeros(maxNb - list(idx1).count(item), dtype=int)]) for item in
         list(set(idx1))])

    if len(values) == 0:
        return maxNb, idxNb
    else:
        outVal = [np.zeros([len(idxNb), maxNb] + list(val.shape)[2:], dtype=val.dtype) for val in values]

        for iVal, val in enumerate(values):
            (outVal[iVal])[idxNb > 0] = val[adjMat > 0]

        return tuple([maxNb, idxNb] + outVal)


def get_nb(Rcart, lattice, dcut):
    Rij = Rcart[None, :, :] - Rcart[:, None, :]
    Rij = cart2frac(frac2cart(Rij, lattice), lattice)

    d2ij = np.sum(Rij ** 2, axis=-1)
    Rij[d2ij > dcut ** 2] = 0

    adjMat = np.array((0 < d2ij) & (d2ij < dcut ** 2), dtype=int)

    maxNb, idxNb, Rij = adjMat2adjList(adjMat, Rij)

    return idxNb, Rij, maxNb


def get_distances(Rij):
    dij = np.sqrt(np.sum(Rij**2, axis=-1))
    dijk = np.sqrt(np.sum((Rij[:, :, None, :] - Rij[:, None, :, :])**2, axis=-1))
    dijk[dij == 0] = 0
    dijk.transpose([0, 2, 1])[dij == 0] = 0
    Rhat = np.zeros_like(Rij)
    Rhat[dij > 0] = Rij[dij > 0] / dij[dij > 0, None]
    return dij, dijk, Rhat