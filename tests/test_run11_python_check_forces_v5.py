import numpy as np
import torch
import torch.nn
import time
from pydoc import locate
import pandas as pd

from chemtorch.nn import get_weights, get_biases, np2torch, torch2np, mlnn, d_mlnn, get_forces, mlnn_optimize
from chemtorch.ml import get_scalar_csv
from chemtorch.io import stream_structures
from chemtorch.io.parameters import load_weights_biases
from chemtorch.features import get_dG_dR


from chemtorch.parameters import settings

settings['Router'] = 6.2

weights, biases = load_weights_biases(len(settings['hidden_layers'])+1)

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

gscalar = get_scalar_csv(settings["valid_feat_file"])
escalar = get_scalar_csv(settings["valid_engy_file"])

dtype, device = locate(settings['dtype']), settings['device']

M2, M3 = settings['M2'], settings['M3']
num_feat, num_engy = M2 + M3 ** 3, 1
mlp = [num_feat, *settings['hidden_layers'], num_engy]
Rc = settings['Router']

# initialize weights and biases if they are not provided
weights = list(np2torch(*weights))
biases = list(np2torch(*biases))
for (w, b) in zip(weights, biases): w.requires_grad, b.requires_grad = True, True

# convert gscalar and escale to torch tensors
gscalar, escalar = list(np2torch(*gscalar, dtype=dtype)), list(np2torch(*escalar, dtype=dtype))


start = time.time()
g, g_dldl, g_dpdl, idxNb = get_dG_dR(Rcart, lattice,
                                     basis='cosine', M2=M2, M3=M3, Rinner=0, Router=Rc)
g, g_dldl, g_dpdl = np2torch(g, g_dldl, g_dpdl, dtype=dtype)
feat_scaled = gscalar[0] * g + gscalar[1]

nn_out2, d_nn_out2 = d_mlnn(feat_scaled, weights, biases, activation="sigmoid")

energies = nn_out2
forces = get_forces(g_dldl, g_dpdl, d_nn_out2, idxNb, gscalar[0])