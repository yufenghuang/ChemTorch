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
from pydoc import locate

from chemtorch.parameters import settings

dtype, device = locate(settings['dtype']), settings['device']

M2, M3 = settings['M2'], settings['M3']
num_feat, num_engy = M2 + M3**3, 1
mlp = [num_feat] + settings['hidden_layers'] + [num_engy]
Rc = settings['Router']
weights, biases = ct_nn.get_weights(mlp), ct_nn.get_biases(mlp)
optimizer = torch.optim.Adam(biases + weights, lr=settings['learning_rate'])

feat_a, feat_b = ct_ml.get_scalar_csv(settings["valid_feat_file"])
engy_a, engy_b = ct_ml.get_scalar_csv(settings["valid_engy_file"])

feat_a, feat_b = ct_nn.np2torch(feat_a, feat_b, dtype=dtype)
engy_a, engy_b = ct_nn.np2torch(engy_a, engy_b, dtype=dtype)


for i_epoch in range(settings['epoch']):
    start = time.time()
    # featFile = "tests\\data\\feat" if platform == 'win32' else "tests/data/feat"
    # engyFile = "tests\\data\\engy" if platform == 'win32' else "tests/data/engy"
    feat_chunk = pd.read_csv(settings['train_feat_file'], header=None, chunksize=settings['chunk_size'])
    engy_chunk = pd.read_csv(settings['train_engy_file'], header=None, chunksize=settings['chunk_size'])
    for step, (feat, engy) in enumerate(zip(feat_chunk, engy_chunk)):

        feat_scaled = feat_a * ct_nn.np2torch(feat.values, dtype=dtype) + feat_b
        engy_scaled = engy_a * ct_nn.np2torch(engy.values.reshape((-1, 1)), dtype=dtype) + engy_b

        nn_out = ct_nn.mlnn(feat_scaled, weights, biases, activation="sigmoid")
        loss = torch.sum((nn_out - engy_scaled)**2)
        ct_nn.mlnn_optimize(loss, optimizer)

    Ei = ct_nn.np2torch(engy.values.reshape((-1, 1)))
    Ep = (nn_out - engy_b)/engy_a
    rmse = torch.sqrt(torch.mean((Ep - Ei)**2))
    print(i_epoch, ct_nn.torch2np(rmse), time.time()-start)


filename = settings['input_file']
pwmat_mmt = ct_io.stream_structures(filename, format=settings['input_format'], get_forces=True)

for iepoch in range(settings['epoch']):
    for step, (Rcart, lattice, atom_types, F, Ei) in enumerate(pwmat_mmt):
        g, g_dldl, g_dpdl, idxNb = ct_ft.get_dG_dR(Rcart, lattice, basis='cosine', M2=M2, M3=M3, Rinner=0, Router=Rc)
        g, g_dldl, g_dpdl = ct_nn.np2torch(g, g_dldl, g_dpdl, dtype=dtype)
        feat_scaled = feat_a * g + feat_b

        nn_out2, d_nn_out2 = ct_nn.d_mlnn(feat_scaled, weights, biases, activation="sigmoid")
        forces = ct_nn.get_forces(g_dldl, g_dpdl, d_nn_out2, idxNb, feat_a)
        z = (nn_out2 - engy_b)/engy_a
        dz = forces / engy_a

        y, dy = ct_nn.np2torch(Ei[:, None], F, dtype=dtype)
        F_mse = torch.mean((dz - dy)**2)
        E_mse = torch.mean((z - y)**2)
        loss = E_mse + F_mse

        ct_nn.mlnn_optimize(loss, optimizer)
        optimizer.step()

        print(step, *ct_nn.torch2np(torch.sqrt(E_mse), torch.sqrt(F_mse)), time.time()-start)
