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
from sklearn.preprocessing import MinMaxScaler

import time

engy_scalar = MinMaxScaler(feature_range=(0,1))
feat_scalar = MinMaxScaler(feature_range=(0,1))

import pandas as pd

dtype, device = torch.float, torch.device('cpu')

M2, M3 = 25, 5
num_feat, num_engy = M2 + M3**3, 1
mlp = [num_feat, 50, 50, 50, num_engy]
weights, biases = get_weights(mlp, xavier=True), get_biases(mlp)
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


for iter in range(50):
    start = time.time()
    feat_chunk = pd.read_csv(featFile, header=None, chunksize=1000)
    engy_chunk = pd.read_csv(engyFile, header=None, chunksize=1000)
    for step, (feat, engy) in enumerate(zip(feat_chunk, engy_chunk)):
        feat = feat.values.astype(np.float32)
        engy = engy.values.astype(np.float32).reshape(-1,1)
        feat_scaled = torch.from_numpy(feat_a * feat + feat_b)
        engy_scaled = torch.from_numpy(engy_a * engy + engy_b)

        nn_out = mlnn(feat_scaled, weights, biases, activation="sigmoid")
        loss = torch.sum((nn_out - engy_scaled)**2)
        mlnn_optimize(loss, optimizer)

    Ep = (nn_out - torch_engy_b)/torch_engy_a
    mse = torch.mean((Ep - torch.from_numpy(engy))**2)
    print(iter, step, mse.data.numpy(), time.time()-start)
