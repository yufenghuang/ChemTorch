import numpy as np
import torch
import torch.nn
import torch.nn.init
from sys import platform
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time
import chemtorch.nn as ct_nn
import chemtorch.io as ct_io
import chemtorch.structure as ct_st
import chemtorch.features.basis.piecewise_cosine as ct_ft_cos

engy_scalar = MinMaxScaler(feature_range=(0,1))
feat_scalar = MinMaxScaler(feature_range=(0,1))

dtype, device = torch.float, torch.device('cpu')

M2, M3 = 25, 5
num_feat, num_engy = M2 + M3**3, 1
mlp = [num_feat, 50, 50, 50, num_engy]
weights, biases = ct_nn.get_weights(mlp, xavier=True), ct_nn.get_biases(mlp)
optimizer = torch.optim.Adam(biases + weights, lr=1e-4)

vfeatFile = "tests\\data\\vfeat" if platform == 'win32' else "tests/data/vfeat"
vengyFile = "tests\\data\\vengy" if platform == 'win32' else "tests/data/vengy"
feat = pd.read_csv(vfeatFile, header=None).values.astype(np.float32)
engy = pd.read_csv(vengyFile, header=None).values.astype(np.float32).reshape(-1, 1)
feat_scalar.fit_transform(feat)
engy_scalar.fit_transform(engy)
(a, b) = feat.shape
feat_b = feat_scalar.transform(np.zeros((1, b))).astype(np.float32)
feat_a = feat_scalar.transform(np.ones((1, b))).astype(np.float32) - feat_b
engy_b = engy_scalar.transform(np.zeros((1,1))).astype(np.float32)
engy_a = engy_scalar.transform(np.ones((1,1))).astype(np.float32) - engy_b

torch_engy_b = torch.from_numpy(engy_b)
torch_engy_a = torch.from_numpy(engy_a)

featFile = "tests\\data\\feat" if platform == 'win32' else "tests/data/feat"
engyFile = "tests\\data\\engy" if platform == 'win32' else "tests/data/engy"

featFile = "tests\\data\\feat" if platform == 'win32' else "tests/data/feat"
engyFile = "tests\\data\\engy" if platform == 'win32' else "tests/data/engy"

for iter in range(10):
    start = time.time()
    feat_chunk = pd.read_csv(featFile, header=None, chunksize=1000)
    engy_chunk = pd.read_csv(engyFile, header=None, chunksize=1000)
    for step, (feat, engy) in enumerate(zip(feat_chunk, engy_chunk)):
        feat = feat.values.astype(np.float32)
        engy = engy.values.astype(np.float32).reshape(-1,1)
        feat_scaled = torch.from_numpy(feat_a * feat + feat_b)
        engy_scaled = torch.from_numpy(engy_a * engy + engy_b)

        nn_out = ct_nn.mlnn(feat_scaled, weights, biases, activation="sigmoid")
        loss = torch.sum((nn_out - engy_scaled)**2)
        ct_nn.mlnn_optimize(loss, optimizer)

    Ep = (nn_out - torch_engy_b)/torch_engy_a
    rmse = torch.sqrt(torch.mean((Ep - torch.from_numpy(engy))**2))
    print(iter, step, rmse.data.numpy(), time.time()-start)


for iter in range(2):
    feat_chunk = pd.read_csv(featFile, header=None, chunksize=256)
    engy_chunk = pd.read_csv(engyFile, header=None, chunksize=256)

    filename = "tests\data\MOVEMENT.train" if platform == 'win32' else "tests/data/MOVEMENT.train"
    mmt = ct_io.read_PWMat_movement(filename, get_forces=True, get_velocities=True, get_Ei=True, get_Epot=True)

    for step, (engy, feat) in enumerate(zip(engy_chunk, feat_chunk)):
        start = time.time()
        n_atoms, lattice, atom_types, Rfrac, F, V, Ei, Epot = next(mmt)

        lattice = ct_st.standardize_lattice(lattice)
        Rcart = ct_st.frac2cart(Rfrac, lattice)
        idxNb, Rij, maxNb = ct_st.get_nb(Rcart, lattice, dcut=6.2)
        dij, dijk, Rhat = ct_st.get_distances(Rij)
        g, g_dldl, g_dpdl = ct_ft_cos.get_d_features(dij, dijk, Rhat, M2, M3, Router=6.2)
        x = torch.from_numpy(g.astype(np.float32))
        dim = Rfrac.shape[-1]

        feat = feat.values.astype(np.float32)
        engy = engy.values.astype(np.float32).reshape(-1,1)
        feat_scaled = torch.from_numpy(feat_a * feat + feat_b)
        engy_scaled = torch.from_numpy(engy_a * engy + engy_b)

        nn_out1 = ct_nn.mlnn(feat_scaled, weights, biases, activation="sigmoid")
        nn_out2, d_nn_out2 = ct_nn.d_mlnn(feat_scaled, weights, biases, activation="sigmoid")

        d_nn_out2 = d_nn_out2 * torch.from_numpy(feat_a)
        fll = torch.squeeze(torch.matmul(d_nn_out2[:, None, :], torch.from_numpy(g_dldl.astype(np.float32))))

        fln = torch.zeros((n_atoms, maxNb, dim))
        fln[np.where(idxNb>0)] = torch.squeeze(torch.matmul(d_nn_out2[idxNb[idxNb>0]-1][:, None, :], torch.from_numpy(g_dpdl[idxNb>0].astype(np.float32))))
        neg_force = torch.sum(fln, dim=1) + fll

        y = torch.from_numpy(Ei[:, None].astype(np.float32))
        dy = torch.from_numpy(F.astype(np.float32))

        z = (nn_out2 - torch_engy_b)/torch_engy_a
        dz = neg_force/torch_engy_a

        dF2 = torch.mean((dz - dy)**2)
        dE2 = torch.mean((z - y)**2)
        loss = dE2 + dF2
        ct_nn.mlnn_optimize(loss, optimizer)
        optimizer.step()
        print(iter, step, np.sqrt(dE2.detach().numpy()), np.sqrt(dF2.detach().numpy()), time.time()-start)
