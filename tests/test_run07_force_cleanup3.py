import numpy as np
import torch
import torch.nn
import torch.nn.init
from sys import platform
import pandas as pd
import time
import chemtorch.nn as ct_nn
import chemtorch.io as ct_io
import chemtorch.ml as ct_ml
import chemtorch.features as ct_ft

dtype, device = torch.float, torch.device('cpu')

M2, M3 = 25, 5
num_feat, num_engy = M2 + M3**3, 1
mlp = [num_feat, 50, 50, num_engy]
weights, biases = ct_nn.get_weights(mlp), ct_nn.get_biases(mlp)
optimizer = torch.optim.Adam(biases + weights, lr=1e-4)
Rc = 6.2

vfeatFile = "tests\\data\\vfeat" if platform == 'win32' else "tests/data/vfeat"
vengyFile = "tests\\data\\vengy" if platform == 'win32' else "tests/data/vengy"
feat_a, feat_b = ct_ml.get_scalar_csv(vfeatFile)
engy_a, engy_b = ct_ml.get_scalar_csv(vengyFile)
torch_engy_a, torch_engy_b = ct_nn.torch2np(engy_a, engy_b, dtype=dtype)


for iter in range(5):
    start = time.time()
    featFile = "tests\\data\\feat" if platform == 'win32' else "tests/data/feat"
    engyFile = "tests\\data\\engy" if platform == 'win32' else "tests/data/engy"
    feat_chunk = pd.read_csv(featFile, header=None, chunksize=1000)
    engy_chunk = pd.read_csv(engyFile, header=None, chunksize=1000)
    for step, (feat, engy) in enumerate(zip(feat_chunk, engy_chunk)):

        feat_scaled = ct_nn.np2torch(feat_a * feat.values + feat_b, dtype=dtype)
        engy_scaled = ct_nn.np2torch(engy_a * engy.values.reshape((-1, 1)) + engy_b, dtype=dtype)

        nn_out = ct_nn.mlnn(feat_scaled, weights, biases, activation="sigmoid")
        loss = torch.sum((nn_out - engy_scaled)**2)
        ct_nn.mlnn_optimize(loss, optimizer)

    Ep = (nn_out - torch_engy_b)/torch_engy_a
    rmse = torch.sqrt(torch.mean((Ep - ct_nn.np2torch(engy.values.reshape((-1, 1))))**2))
    print(iter, ct_nn.torch2np(rmse), time.time()-start)


filename = "tests\data\MOVEMENT.train" if platform == 'win32' else "tests/data/MOVEMENT.train"
pwmat_mmt = ct_io.stream_structures(filename, format="pwmat", get_forces=True)
for step, (Rcart, lattice, atom_types, F, Ei) in enumerate(pwmat_mmt):
    g, g_dldl, g_dpdl, idxNb = ct_ft.get_dG_dR(Rcart, lattice, basis='cosine', M2=M2, M3=M3, Rinner=0, Router=Rc)
    feat_scaled = feat_a * g + feat_b
    g_dldl, g_dpdl, torch_feat_a, feat_scaled = ct_nn.np2torch(g_dldl, g_dpdl, feat_a, feat_scaled, dtype=dtype)
    nn_out2, d_nn_out2 = ct_nn.d_mlnn(feat_scaled, weights, biases, activation="sigmoid")
    forces = ct_nn.get_forces(g_dldl, g_dpdl, d_nn_out2, idxNb, torch_feat_a)
    z = (nn_out2 - torch_engy_b)/torch_engy_a
    dz = forces / torch_engy_a

    y, dy = ct_nn.np2torch(Ei[:, None], F, dtype=dtype)
    F_mse = torch.mean((dz - dy)**2)
    E_mse = torch.mean((z - y)**2)
    loss = E_mse + F_mse

    ct_nn.mlnn_optimize(loss, optimizer)
    optimizer.step()

    print(step, *ct_nn.torch2np(torch.sqrt(E_mse), torch.sqrt(F_mse)), time.time()-start)
