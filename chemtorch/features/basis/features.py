from chemtorch.structure.lattice import standardize_lattice
from chemtorch.structure.coordinates import frac2cart, get_nb, get_distances
import chemtorch.features.basis.piecewise_cosine as p_cosine

def get_cos_features():
    pass

def get_gaussian_feats():
    pass

def get_dcosFeat_dR(Rfrac, lattice, M2, M3, Rinner=0.0, Router=6):
    lattice = standardize_lattice(lattice)
    Rcart = frac2cart(Rfrac, lattice)
    idxNb, Rij, maxNb = get_nb(Rcart, lattice, dcut=6.2)
    dij, dijk, Rhat = get_distances(Rij)
    g, g_dldl, g_dpdl = p_cosine.get_d_features(dij, dijk, Rhat, M2, M3, Rinner=Rinner, Router=Router)
    return g, g_dldl, g_dpdl

