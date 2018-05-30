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


