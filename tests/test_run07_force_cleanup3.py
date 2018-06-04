import numpy as np
import torch
import torch.nn
import torch.nn.init
from sys import platform
import pandas as pd
import time
import chemtorch.nn as ct_nn
import chemtorch.io as ct_io
import chemtorch.structure as ct_st
import chemtorch.features.basis.piecewise_cosine as ct_ft_cos
import chemtorch.ml as ct_ml
import chemtorch.features as ct_ft

dtype, device = torch.float, torch.device('cpu')

M2, M3 = 25, 5
num_feat, num_engy = M2 + M3**3, 1
mlp = [num_feat, 50, 50, 50, num_engy]
weights, biases = ct_nn.get_weights(mlp), ct_nn.get_biases(mlp)
optimizer = torch.optim.Adam(biases + weights, lr=1e-4)
Rc = 6.2

vfeatFile = "tests\\data\\vfeat" if platform == 'win32' else "tests/data/vfeat"
vengyFile = "tests\\data\\vengy" if platform == 'win32' else "tests/data/vengy"

feat_a, feat_b = ct_ml.get_scalar_csv(vfeatFile)
engy_a, engy_b = ct_ml.get_scalar_csv(vengyFile)

torch_engy_b = torch.from_numpy(engy_b).to(dtype)
torch_engy_a = torch.from_numpy(engy_a).to(dtype)

featFile = "tests\\data\\feat" if platform == 'win32' else "tests/data/feat"
engyFile = "tests\\data\\engy" if platform == 'win32' else "tests/data/engy"


for iter in range(1):
    start = time.time()
    feat_chunk = pd.read_csv(featFile, header=None, chunksize=1000)
    engy_chunk = pd.read_csv(engyFile, header=None, chunksize=1000)
    for step, (feat, engy) in enumerate(zip(feat_chunk, engy_chunk)):

        feat = feat.values
        engy = engy.values.reshape(-1,1)

        feat_scaled = torch.from_numpy(feat_a * feat + feat_b).to(dtype)
        engy_scaled = torch.from_numpy(engy_a * engy + engy_b).to(dtype)

        nn_out = ct_nn.mlnn(feat_scaled, weights, biases, activation="sigmoid")
        loss = torch.sum((nn_out - engy_scaled)**2)
        ct_nn.mlnn_optimize(loss, optimizer)

    Ep = (nn_out - torch_engy_b)/torch_engy_a
    rmse = torch.sqrt(torch.mean((Ep - torch.from_numpy(engy).to(dtype))**2))
    print(iter, step, rmse.data.numpy(), time.time()-start)


for iter in range(2):
    feat_chunk = pd.read_csv(featFile, header=None, chunksize=256)
    engy_chunk = pd.read_csv(engyFile, header=None, chunksize=256)

    filename = "tests\data\MOVEMENT.train" if platform == 'win32' else "tests/data/MOVEMENT.train"
    pwmat_mmt = ct_io.stream_structures(filename, format="pwmat", get_forces=True)

    for step, (engy, feat) in enumerate(zip(engy_chunk, feat_chunk)):
        start = time.time()

        Rcart, lattice, atom_types, F, Ei = next(pwmat_mmt)
        n_atoms, dim = Rcart.shape
        engy = Ei.reshape((-1,1))

        idxNb, Rij, maxNb = ct_st.get_nb(Rcart, lattice, dcut=6.2)

        g, g_dldl, g_dpdl = ct_ft.get_dG_dR(Rcart, lattice, basis='cosine', M2=M2, M3=M3, Rinner=0, Router=Rc)

        feat = feat.values.astype(np.float32)
        engy = engy.values.astype(np.float32).reshape(-1,1)
        feat_scaled = torch.from_numpy(feat_a * feat + feat_b).to(dtype)
        engy_scaled = torch.from_numpy(engy_a * engy + engy_b).to(dtype)

        nn_out2, d_nn_out2 = ct_nn.d_mlnn(feat_scaled, weights, biases, activation="sigmoid")

        d_nn_out2 = d_nn_out2 * torch.from_numpy(feat_a).to(dtype)
        fll = torch.squeeze(torch.matmul(d_nn_out2[:, None, :], torch.from_numpy(g_dldl.astype(np.float32))))

        fln = torch.zeros((n_atoms, maxNb, dim))
        fln[np.where(idxNb>0)] = torch.squeeze(torch.matmul(d_nn_out2[idxNb[idxNb>0]-1][:, None, :], torch.from_numpy(g_dpdl[idxNb>0].astype(np.float32))))
        neg_force = torch.sum(fln, dim=1) + fll

        y = torch.from_numpy(Ei[:, None].astype(np.float32))
        dy = torch.from_numpy(F.astype(np.float32))

        z = (nn_out2 - torch_engy_b)/torch_engy_a
        dz = neg_force/torch_engy_a

        F_mse = torch.mean((dz - dy)**2)
        E_mse = torch.mean((z - y)**2)

        loss = E_mse + F_mse
        ct_nn.mlnn_optimize(loss, optimizer)
        optimizer.step()
        print(iter, step, np.sqrt(E_mse.detach().numpy()), np.sqrt(F_mse.detach().numpy()), time.time()-start)
