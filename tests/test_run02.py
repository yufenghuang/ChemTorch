# Generate features

import numpy as np
import torch
import torch.nn
import torch.nn.init
from chemtorch.structure.coordinates import frac2cart
from chemtorch.structure.lattice import standardize_lattice
from chemtorch.nn.common import get_biases, get_weights, mlnn, mlnn_optimize
from chemtorch.io.read import read_PWMat_movement
from sys import platform
from chemtorch.features.basis.piecewise_cosine import get_d_features
from chemtorch.structure.coordinates import get_nb, get_distances

import pandas as pd

dtype, device = torch.float, torch.device('cpu')

M2, M3 = 25, 5
num_feat, num_engy = M2 + M3**3, 1
mlp = [num_feat, 50, 50, 50, num_engy]
weights, biases = get_weights(mlp, xavier=True), get_biases(mlp)
optimizer = torch.optim.Adam(biases + weights, lr=1e-4)

featFile = "tests\\data\\vfeat" if platform == 'win32' else "tests/data/vfeat"
engyFile = "tests\\data\\vengy" if platform == 'win32' else "tests/data/vengy"
filename = "tests\data\MOVEMENT.valid" if platform == 'win32' else "tests/data/MOVEMENT.valid"
mmt = read_PWMat_movement(filename, get_forces=True, get_velocities=True, get_Ei=True, get_Epot=True)

for iteration, (n_atoms, lattice, atom_types, Rfrac, F, V, Ei, Epot) in enumerate(mmt):
    lattice = standardize_lattice(lattice)
    Rcart = frac2cart(Rfrac, lattice)
    idxNb, Rij, maxNb = get_nb(Rcart, lattice, dcut=6.2)
    dij, dijk, Rhat = get_distances(Rij)
    g, g_dldl, g_dpdl = get_d_features(dij, dijk, Rhat, M2, M3, Router=6.2)

    x = torch.from_numpy(g.astype(np.float32))
    y = torch.from_numpy(Ei[:, None].astype(np.float32))

    print(iteration)

    pd.DataFrame(g.astype(np.float32)).to_csv(featFile, header=False, index=False, mode="a")
    pd.DataFrame(Ei[:, None].astype(np.float32)).to_csv(engyFile, header=False, index=False, mode="a")

# nn_out = mlnn(x, weights, biases, activation="sigmoid")
# loss = torch.sum((nn_out - y)**2)
# mlnn_optimize(loss, optimizer)
