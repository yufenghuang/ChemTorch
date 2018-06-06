from chemtorch.nn.coord2energy import pcosine, dpcosine
from chemtorch.nn import torch2np, np2torch
from chemtorch.parameters import settings
from chemtorch.io import stream_structures
from chemtorch.structure import get_nblist
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from torch.autograd import grad

from chemtorch.structure.graph import adj_mat2list, adj_list2mat


def cart2frac(Rcart, lattice):
    """
    Convert cartesian coordinates to fractional coordinates
    :param R_cart: cartesian coordinates
    :param lattice: lattice vector with shape n x n for n-dimensional system
    :return: fractional coordinate, for n dimensinoal system, \
             the first n-columns are assumed to be fractional
    """
    Rfrac = Rcart.clone()
    if 0 < len(lattice):
        X, LU = torch.gesv(Rcart[:, :len(lattice)].t(), lattice.t())
        Rfrac[:, :len(lattice)] = X.t()
        Rfrac[:, :len(lattice)] = Rfrac[:, :len(lattice)] - torch.floor(Rfrac[:, :len(lattice)])
    return Rfrac


def frac2cart(Rfrac, lattice):
    """
    Convert fractional coordinates to cartesian coordinates
    :param R_frac: input fractional coordinate, for n dimensinoal system, \
                   the first n-columns are assumed to be fractional
    :param lattice: lattice vector with shape n x n for n-dimensional system
    :return: R_cart: cartesian coordinates
    """
    Rcart = Rfrac.clone()
    if 0 < len(lattice):
        Rcart[:, :len(lattice)] = torch.mm(Rcart[:, :len(lattice)], lattice)
    return Rcart


def get_nb(Rcart, lattice, dcut):

    nAtoms = Rcart.shape[0]
    dim = Rcart.shape[-1]

    Rij = Rcart[None, :, :] - Rcart[:, None, :]

    Rfrac = cart2frac(Rij.reshape((-1, dim)), lattice)
    Rfrac[Rfrac>0.5] = Rfrac[Rfrac>0.5]-1
    Rij = frac2cart(Rfrac, lattice).reshape((nAtoms, -1, dim))

    d2ij = torch.sum(Rij ** 2, dim=-1)
    Rij[d2ij > dcut ** 2] = 0

    # adjMat = np.array((0 < d2ij) & (d2ij < dcut ** 2), dtype=int)

    adjMat = (0 < d2ij) & (d2ij < dcut **2)

    nblist = torch.from_numpy(adj_mat2list(adjMat.numpy()))

    Rij_truncated = torch.zeros(*nblist.shape, dim)
    Rij_truncated[nblist > 0] = Rij[adjMat == 1]

    return nblist, Rij_truncated


def get_distances(Rij):
    dij = torch.sqrt(torch.sum(Rij**2, dim=-1))
    dijk = torch.sqrt(torch.sum((Rij[:, :, None, :] - Rij[:, None, :, :])**2, dim=-1))
    dijk[dij == 0] = 0
    dijk.transpose(1,2)[dij == 0] = 0
    Rhat = torch.zeros_like(Rij)
    Rhat[dij > 0] = Rij[dij > 0] / (dij[dij > 0])[:, None]
    return dij, dijk, Rhat

pwmat_mmt = stream_structures(settings['input_file'], format=settings['input_format'], get_forces=True)
Rcart, lattice, atom_types, F, Ei = next(pwmat_mmt)

Rcart, lattice = np2torch(Rcart, lattice)

nblist, Rij = get_nb(Rcart, lattice, settings['Router'])
dij, dijk, Rhat = get_distances(Rij)

# nblist, nbmat = get_nblist(Rcart, lattice, dcut=settings['Router'])
# natoms, maxNb = nblist.shape
#
# nblist = torch.from_numpy(nblist)
# nbmat = torch.from_numpy(nbmat)
#
# nblist2 = torch.from_numpy(adj_mat2list(nbmat.numpy()))
# nbmat2 = torch.from_numpy(adj_list2mat(nblist.numpy()))
#
# dim = Rcart.shape[-1]
#
# lattice = torch.from_numpy(lattice)
# lattice = lattice.to(torch.float)
# Rcart = torch.from_numpy(Rcart)
# Rcart = Rcart.to(torch.float)
# Rcart.requires_grad=True
#
# Rfrac = cart2frac(Rcart, lattice)
# Rcart2 = frac2cart(Rfrac, lattice)
#
# Rij_mat = Rcart[None, :, :] - Rcart[:, None, :]
# Rij = torch.zeros((natoms, maxNb, dim))
# Rij[nblist > 0] = Rij_mat[nbmat == 1]
#
# Dij, Dijk, Rh = get_distances(Rij)
#
