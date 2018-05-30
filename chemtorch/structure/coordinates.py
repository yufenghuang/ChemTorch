import numpy as np


def cart2frac(R_cart, lattice):
    R_frac = R_cart.copy()
    if len(lattice) >= 2:
        R_frac[:, :len(lattice)] = np.linalg.solve(lattice.T, R_cart[:, :len(lattice)].T).T
    elif len(lattice) == 1:
        R_frac[:, 0] = R_cart[:, 0] / lattice.reshape(1)
    R_frac[:, :len(lattice)] = R_frac[:, :len(lattice)] - np.floor(R_frac[:, :len(lattice)])
    return R_frac


def frac2cart(R_frac, lattice):
    R_cart = R_frac.copy()
    if len(lattice) >= 2:
        R_cart[:, :len(lattice)] = np.dot(R_cart[:, :len(lattice)], lattice)
    elif len(lattice) == 1:
        R_cart[:, 0] = R_cart[:, 0] * lattice.reshape(1)
    return R_cart



