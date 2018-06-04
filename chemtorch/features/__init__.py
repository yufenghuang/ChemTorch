from .basis import piecewise_cosine as cos_basis
from .basis import gaussian_like as g_basis
from .. import structure


def get_dcos_dR(Rcart, lattice, M2, M3, Rinner=0.0, Router=6):
    lattice = structure.standardize_lattice(lattice)
    # Rcart = structure.frac2cart(Rfrac, lattice)
    idxNb, Rij, maxNb = structure.get_nb(Rcart, lattice, dcut=6.2)
    dij, dijk, Rhat = structure. get_distances(Rij)
    g, g_dldl, g_dpdl = cos_basis.get_d_features(dij, dijk, Rhat, M2, M3, Rinner=Rinner, Router=Router)
    return g, g_dldl, g_dpdl


def get_dGaussian_dR():
    pass


def get_dG_dR(Rcart, lattice, basis='cosine', **basis_params):
    if basis == "cosine":
        cos_params = ["M2", "M3", "Rinner", "Router"]
        for param in cos_params:
            assert param in basis_params, \
                   "Parameter " + param + " is not provided!"
            if param in ["M2", "M3"]:
                basis_params[param] = int(basis_params[param])
            else:
                basis_params[param] = float(basis_params[param])
        return get_dcos_dR(Rcart, lattice, basis_params['M2'], basis_params['M3'],
                           basis_params['Rinner'], basis_params['Router'])
    else:
        pass


