import numpy as np
import torch
import torch.nn
import time
from pydoc import locate
import pandas as pd

from chemtorch.nn import get_weights, get_biases, np2torch, torch2np, mlnn, d_mlnn, get_forces, mlnn_optimize
from chemtorch.ml import get_scalar_csv
from chemtorch.io import stream_structures
from chemtorch.features import get_dG_dR

from chemtorch.parameters import settings
settings['epoch'] = 5

weights, biases = None, None
dtype, device = locate(settings['dtype']), settings['device']

M2, M3 = settings['M2'], settings['M3']
num_feat, num_engy = M2 + M3 ** 3, 1
mlp = [num_feat] + settings['hidden_layers'] + [num_engy]

# initialize weights and biases if they are not provided
weights = get_weights(mlp) if weights is None else list(np2torch(*weights))
biases = get_biases(mlp) if biases is None else list(np2torch(*biases))
for (w, b) in zip(weights, biases): w.requires_grad, b.requires_grad = True, True

optimizer = torch.optim.Adam(biases + weights, lr=settings['learning_rate'])

feat_a, feat_b = get_scalar_csv(settings["valid_feat_file"])
engy_a, engy_b = get_scalar_csv(settings["valid_engy_file"])

feat_a, feat_b = np2torch(feat_a, feat_b, dtype=dtype)
engy_a, engy_b = np2torch(engy_a, engy_b, dtype=dtype)

for i_epoch in range(settings['epoch']):

    start = time.time()

    feat_chunk = pd.read_csv(settings['train_feat_file'], header=None, chunksize=settings['chunk_size'])
    engy_chunk = pd.read_csv(settings['train_engy_file'], header=None, chunksize=settings['chunk_size'])

    for step, (feat, engy) in enumerate(zip(feat_chunk, engy_chunk)):
        feat_scaled = feat_a * np2torch(feat.values, dtype=dtype) + feat_b
        engy_scaled = engy_a * np2torch(engy.values.reshape((-1, 1)), dtype=dtype) + engy_b

        nn_out = mlnn(feat_scaled, weights, biases, activation="sigmoid")
        loss = torch.sum((nn_out - engy_scaled) ** 2)
        mlnn_optimize(loss, optimizer)

    Ei = np2torch(engy.values.reshape((-1, 1)))
    Ep = (nn_out - engy_b) / engy_a

    rmse = torch.sqrt(torch.mean((Ep - Ei) ** 2))

    print(i_epoch, torch2np(rmse), time.time() - start)

dtype, device = locate(settings['dtype']), settings['device']

M2, M3 = settings['M2'], settings['M3']
num_feat, num_engy = M2 + M3 ** 3, 1
mlp = [num_feat] + settings['hidden_layers'] + [num_engy]
Rc = settings['Router']

# initialize weights and biases if they are not provided
# weights = get_weights(mlp) if weights is None else list(np2torch(*weights))
# biases = get_biases(mlp) if biases is None else list(np2torch(*biases))
# for (w, b) in zip(weights, biases): w.requires_grad, b.requires_grad = True, True
for (w, b) in zip(weights, biases): print(len(w), w.reshape(-1)[0], len(b), b.reshape(-1)[0])

optimizer = torch.optim.Adam(biases + weights, lr=settings['learning_rate'])

feat_a, feat_b = get_scalar_csv(settings["valid_feat_file"])
engy_a, engy_b = get_scalar_csv(settings["valid_engy_file"])

feat_a, feat_b = np2torch(feat_a, feat_b, dtype=dtype)
engy_a, engy_b = np2torch(engy_a, engy_b, dtype=dtype)

filename = settings['input_file']
pwmat_mmt = stream_structures(filename, format=settings['input_format'], get_forces=True)

for iepoch in range(settings['epoch']):
    for step, (Rcart, lattice, atom_types, F, Ei) in enumerate(pwmat_mmt):
        start = time.time()
        g, g_dldl, g_dpdl, idxNb = get_dG_dR(Rcart, lattice, basis='cosine', M2=M2, M3=M3, Rinner=0,
                                                   Router=Rc)
        g, g_dldl, g_dpdl = np2torch(g, g_dldl, g_dpdl, dtype=dtype)
        feat_scaled = feat_a * g + feat_b

        nn_out2, d_nn_out2 = d_mlnn(feat_scaled, weights, biases, activation="sigmoid")
        forces = get_forces(g_dldl, g_dpdl, d_nn_out2, idxNb, feat_a)
        z = (nn_out2 - engy_b) / engy_a
        dz = forces / engy_a

        y, dy = np2torch(Ei[:, None], F, dtype=dtype)
        F_mse = torch.mean((dz - dy) ** 2)
        E_mse = torch.mean((z - y) ** 2)
        loss = E_mse + F_mse

        mlnn_optimize(loss, optimizer)
        optimizer.step()

        print(step, *torch2np(torch.sqrt(E_mse), torch.sqrt(F_mse)), time.time() - start)

