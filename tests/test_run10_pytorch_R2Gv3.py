from chemtorch.io import stream_structures
from chemtorch.parameters import settings
from chemtorch.nn.coord2energy import get_distances, get_nb, pcosine, dpcosine
from chemtorch.nn import np2torch, torch2np
import torch
import numpy as np
from torch.autograd import grad

import chemtorch.structure as cs
import chemtorch.features.basis.piecewise_cosine as cf

settings['Router'] = 6.2


M2 = settings['M2']
M3 = settings['M3']
Rinner = settings['Rinner']
Router = settings['Router']
R2 = Router - (Router - Rinner) / M2
R3 = Router - (Router - Rinner) / M3


pwmat_mmt = stream_structures(settings['input_file'], format=settings['input_format'], get_forces=True)
Rcart, lattice, atom_types, F, Ei = next(pwmat_mmt)
nAtoms = len(Rcart)

npRcart = Rcart.copy()
nplattice = lattice.copy()
npIdxNb, npRij, npMaxNb = cs.get_nb(npRcart, nplattice, settings['Router'])
npDij, npDijk, npRhat = cs.get_distances(npRij)

npPhi2 = np.zeros((*npDij.shape, M2))
npPhi3_ij = np.zeros((*npDij.shape, M3))
npPhi3_ijk = np.zeros((*npDijk.shape, M3))

npPhi2[npDij > 0] = cf.piecewise_cosine(npDij[npDij > 0], M2, start=Rinner, stop=R2)
npPhi3_ij[npDij > 0] = cf.piecewise_cosine(npDij[npDij > 0], M3, start=Rinner, stop=R3)
npPhi3_ijk[npDijk > 0] = cf.piecewise_cosine(npDijk[npDijk > 0], M3, start=Rinner, stop=R3)

z3a = np.matmul(npPhi3_ij.transpose([0, 2, 1]), npPhi3_ijk.reshape((nAtoms, npMaxNb, -1)))
z3a = z3a.reshape((nAtoms, M3, npMaxNb, M3)).transpose([0, 2, 1, 3])
g2 = npPhi2.sum(axis=1)
g3 = np.matmul(npPhi3_ij.transpose([0, 2, 1]), z3a.reshape((nAtoms, npMaxNb, -1))).reshape((nAtoms, -1))
g = np.concatenate([g2, g3], axis=1)

# ===============================================================

Rcart, lattice = np2torch(Rcart, lattice)
Rcart.requires_grad = True

nblist, Rij = get_nb(Rcart, lattice, settings['Router'])
dij, dijk, Rhat = get_distances(Rij)


phi2b = torch.zeros(*dij.shape, settings['M2'])
phi3b_ij = torch.zeros(*dij.shape, settings['M3'])
phi3b_ijk = torch.zeros(*dijk.shape, settings['M3'])


phi2b[dij > 0] = pcosine(dij[dij > 0], settings['M2'], start=settings['Rinner'], stop=R2)
phi3b_ij[dij > 0] = pcosine(dij[dij > 0], settings['M3'], start=settings['Rinner'], stop=R3)
phi3b_ijk[dijk > 0] = pcosine(dijk[dijk > 0], settings['M3'], start=settings['Rinner'], stop=R3)


G2 = torch.sum(phi2b, dim=1)

# shape of G3 will be: natoms x alpha x gamma x maxNb
G3 = torch.matmul(phi3b_ij.transpose(1,2)[:, :, None, None, :],
                  phi3b_ijk.transpose(1,3)[:, None, :, :, :]).squeeze()

# shape of G3 will be: natoms x alpha x beta x gamma
G3 = torch.matmul(phi3b_ij.transpose(1,2)[:, None, :, None, :],
                  G3.transpose(2,3)[:, :, None, :, :]).squeeze()

G = torch.cat((G2, G3.reshape(len(G3), -1)), 1)

print("Max abs difference in dij:", np.max(np.abs(npDij - dij.detach().numpy())))
print("Max abs difference in dijk:", np.max(np.abs(npDijk - dijk.detach().numpy())))
print("Max abs difference in Rhat:", np.max(np.abs(npRhat - Rhat.detach().numpy())))
print("Max abs difference in phi2b:", np.max(np.abs(npPhi2 - phi2b.detach().numpy())))
print("Max abs difference in phi3b_ij:", np.max(np.abs(npPhi3_ij - phi3b_ij.detach().numpy())))
print("Max abs difference in phi3b_ijk:", np.max(np.abs(npPhi3_ijk - phi3b_ijk.detach().numpy())))
print("Max abs difference in G2:", np.max(np.abs(g2 - G2.detach().numpy())))
print("Max abs difference in G3:", np.max(np.abs(g3 - G3.detach().reshape(len(G3), -1).numpy())))
print("Max abs difference in G:", np.max(np.abs(g - G.detach().numpy())))