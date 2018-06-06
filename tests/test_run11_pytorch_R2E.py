from chemtorch.io import stream_structures
from chemtorch.parameters import settings
from chemtorch.nn.coord2energy import get_distances, get_nb, pcosine, dpcosine, cart2frac, frac2cart
from chemtorch.nn import np2torch, torch2np, mlnn, get_weights, get_biases
import torch
from chemtorch.structure.graph import adj_mat2list, adj_list2mat
from chemtorch.ml import get_scalar_csv
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
dcut = settings['Router']


pwmat_mmt = stream_structures(settings['input_file'], format=settings['input_format'], get_forces=True)
Rcart, lattice, atom_types, F, Ei = next(pwmat_mmt)
nAtoms = len(Rcart)

Rcart, lattice = np2torch(Rcart, lattice)
Rcart.requires_grad = True

# nAtoms = Rcart.shape[0]
# dim = Rcart.shape[-1]
#

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

M2, M3 = settings['M2'], settings['M3']
num_feat, num_engy = M2 + M3 ** 3, 1
mlp = [num_feat, *settings['hidden_layers'], num_engy]

# initialize weights and biases if they are not provided
weights = get_weights(mlp)
biases = get_biases(mlp)

gscalar = get_scalar_csv(settings["valid_feat_file"])
escalar = get_scalar_csv(settings["valid_engy_file"])
gscalar, escalar = list(np2torch(*gscalar)), list(np2torch(*escalar))

G = gscalar[0] * G + gscalar[1]

E = mlnn(G, weights, biases)

F = grad(torch.sum(E), Rcart, create_graph=True)[0]