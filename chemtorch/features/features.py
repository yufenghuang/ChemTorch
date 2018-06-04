# from chemtorch.structure.lattice import standardize_lattice
# from chemtorch.structure.coordinates import frac2cart, get_nb, get_distances
# import chemtorch.features.basis.piecewise_cosine as p_cosine
from .basis import piecewise_cosine as p_cosine
from ..structure.lattice import standardize_lattice
from ..structure.coordinates import frac2cart, get_nb, get_distances

def get_cos_features():
    pass


def get_gaussian_feats():
    pass


def get_dcos_dR(Rfrac, lattice, M2, M3, Rinner=0.0, Router=6):
    lattice = standardize_lattice(lattice)
    Rcart = frac2cart(Rfrac, lattice)
    idxNb, Rij, maxNb = get_nb(Rcart, lattice, dcut=6.2)
    dij, dijk, Rhat = get_distances(Rij)
    g, g_dldl, g_dpdl = p_cosine.get_d_features(dij, dijk, Rhat, M2, M3, Rinner=Rinner, Router=Router)
    return g, g_dldl, g_dpdl

def get_dGaussian_dR():
    pass


def get_dG_dR(Rfrac, lattice, basis='cosine', **basis_params):
    if basis == "cosine":
        cos_params = ["M2", "M3", "Rinner", "Router"]
        for param in cos_params:
            assert param in basis_params, \
                   "Parameter " + param + " is not provided!"
            basis_params[param] = float(basis_params[param])
        return get_dcos_dR(Rfrac, lattice, basis_params['M2'], basis_params['M3'],
                           basis_params['Rinner'], basis_params['Router'])
    else:
        pass


