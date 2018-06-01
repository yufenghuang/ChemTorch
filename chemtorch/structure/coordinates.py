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
    # if len(lattice) >= 2:
    #     R_frac[:, :len(lattice)] = np.linalg.solve(lattice.T, R_cart[:, :len(lattice)].T).T
    # elif len(lattice) == 1:
    #     R_frac[:, 0] = R_cart[:, 0] / lattice.reshape(1)
    # R_frac[:, :len(lattice)] = R_frac[:, :len(lattice)] - np.floor(R_frac[:, :len(lattice)])
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



