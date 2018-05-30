import numpy as np


def standardize_lattice(lattice):
    L = np.array(lattice)
    L = L.reshape(-1)
    if len(L) == 0 or len(L) == 1:
        return L
    elif len(L) == 4:
        return L.reshape((2, 2))
    elif len(L) == 9:
        return L.reshape((3, 3))
    else:
        raise ValueError("Could not standardize lattice because it has a wrong struture!")


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




