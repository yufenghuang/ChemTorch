import torch
import numpy as np
import math
from ..structure import adj_list2mat, adj_mat2list


def get_forces(dGl_dRl, dGp_dRl, d_nn_out2, idxNb, feat_a):

    n_atoms, maxNb = idxNb.shape
    dim = dGl_dRl.shape[-1]

    d_nn_out2 = d_nn_out2 * feat_a
    fll = torch.matmul(d_nn_out2[:, None, :], dGl_dRl).squeeze()

    fln = torch.zeros((n_atoms, maxNb, dim))
    fln[np.where(idxNb > 0)] = torch.matmul(d_nn_out2[idxNb[idxNb > 0] - 1][:, None, :],
                                            dGp_dRl[np.where(idxNb > 0)]).squeeze()
    forces = -torch.sum(fln, dim=1) - fll

    return forces


dG_to_force = get_forces


def pcosine(Rij, num_basis, start=-1.0, stop=1.0):
    """ Piecewise cosine functions on an array xin, \
        return a matrix of dimension dim(xin)*num_basis"""

    assert Rij.dim() <= 1, "Rij should be a 0- or 1-dimensional array, " \
                              "but the shape of xin is " + str(Rij.shape)
    assert num_basis >= 2, "num_basis has to be greater than 2"
    num_basis = int(num_basis)

    h = (stop - start)/(num_basis-1)
    nodes = torch.linspace(start, stop, num_basis)

    y = Rij[:, None] - nodes
    out = torch.zeros_like(y)

    out[torch.abs(y) < h] = torch.cos(y[torch.abs(y) < h] * math.pi / h)/2 + 1/2

    return out


def dpcosine(Rij, num_basis, start=-1, stop=1):
    """ Derivative of the piecewise cosine functions on an array xin, \
        return a matrix of dimension dim(xin)*num_basis"""

    assert Rij.dim() <= 1, "Rij should be a 0- or 1-dimensional array, " \
                              "but the shape of xin is " + str(Rij.shape)
    assert num_basis >= 2, "num_basis has to be greater than 2"
    num_basis = int(num_basis)

    h = (stop - start)/(num_basis-1)
    nodes = torch.linspace(start, stop, num_basis)

    y = Rij[:, None] - nodes
    out = torch.zeros_like(y)

    out[torch.abs(y) < h] = -1/2 * math.pi / h * torch.sin(y[torch.abs(y) < h] * math.pi / h)

    return out


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
        Rcart[:, :len(lattice)] = torch.matmul(Rfrac[:, :len(lattice)], lattice)
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

    dijk2 = ((Rij[:, :, None, :] - Rij[:, None, :, :]) ** 2).clone()
    dnot0 = (dijk2[:, :, :, 0] + dijk2[:, :, :, 1] + dijk2[:, :, :, 2]) > 0
    w = torch.sqrt(torch.sum(dijk2[dnot0], dim=-1))
    dijk = torch.zeros(*dij.shape, dij.shape[-1])
    dijk[dnot0] = w

    # dijk = torch.sqrt(torch.sum((Rij[:, :, None, :] - Rij[:, None, :, :])**2, dim=-1)).clone()
    # dijk[dij == 0] = 0
    # dijk.transpose(1,2)[dij == 0] = 0
    Rhat = torch.zeros_like(Rij)
    Rhat[dij > 0] = Rij[dij > 0] / (dij[dij > 0])[:, None]
    return dij, dijk, Rhat


