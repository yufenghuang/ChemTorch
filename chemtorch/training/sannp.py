import numpy as np
import torch
import torch.nn
import time
from pydoc import locate
import pandas as pd

from ..nn import get_weights, get_biases, np2torch, torch2np, mlnn, d_mlnn, get_forces, mlnn_optimize
from ..io import stream_structures
from ..features import get_dG_dR


def train_energy(settings, weights=None, biases=None, gscalar=list((1, 0)), escalar=list((1, 0))):
    dtype, device = locate(settings['dtype']), settings['device']

    M2, M3 = settings['M2'], settings['M3']
    num_feat, num_engy = M2 + M3 ** 3, 1
    mlp = [num_feat, *settings['hidden_layers'], num_engy]

    # initialize weights and biases if they are not provided
    weights = get_weights(mlp) if weights is None else list(np2torch(*weights))
    biases = get_biases(mlp) if biases is None else list(np2torch(*biases))
    for (w, b) in zip(weights, biases): w.requires_grad, b.requires_grad = True, True

    # convert gscalar and escale to torch tensors
    gscalar, escalar = list(np2torch(*gscalar, dtype=dtype)), list(np2torch(*escalar, dtype=dtype))

    optimizer = torch.optim.Adam(biases + weights, lr=settings['learning_rate'])

    for i_epoch in range(settings['epoch']):

        start = time.time()

        feat_chunk = pd.read_csv(settings['train_feat_file'], header=None, chunksize=settings['chunk_size'])
        engy_chunk = pd.read_csv(settings['train_engy_file'], header=None, chunksize=settings['chunk_size'])

        for step, (feat, engy) in enumerate(zip(feat_chunk, engy_chunk)):
            feat_scaled = gscalar[0] * np2torch(feat.values, dtype=dtype) + gscalar[1]
            engy_scaled = escalar[0] * np2torch(engy.values.reshape((-1, 1)), dtype=dtype) + escalar[1]

            nn_out = mlnn(feat_scaled, weights, biases, activation="sigmoid")
            loss = torch.sum((nn_out - engy_scaled) ** 2)
            mlnn_optimize(loss, optimizer)

        Ei = np2torch(engy.values.reshape((-1, 1)))
        Ep = (nn_out - escalar[1]) / escalar[0]

        rmse = torch.sqrt(torch.mean((Ep - Ei) ** 2))

        print(i_epoch, torch2np(rmse), time.time() - start)

    return list(torch2np(*weights)), list(torch2np(*biases))


def train_force(settings, weights=None, biases=None, gscalar=list((1, 0)), escalar=list((1, 0))):
    dtype, device = locate(settings['dtype']), settings['device']

    M2, M3 = settings['M2'], settings['M3']
    num_feat, num_engy = M2 + M3 ** 3, 1
    mlp = [num_feat, *settings['hidden_layers'], num_engy]
    Rc = settings['Router']

    # initialize weights and biases if they are not provided
    weights = get_weights(mlp) if weights is None else list(np2torch(*weights))
    biases = get_biases(mlp) if biases is None else list(np2torch(*biases))
    for (w, b) in zip(weights, biases): w.requires_grad, b.requires_grad = True, True

    # convert gscalar and escale to torch tensors
    gscalar, escalar = list(np2torch(*gscalar, dtype=dtype)), list(np2torch(*escalar, dtype=dtype))

    optimizer = torch.optim.Adam(biases + weights, lr=settings['learning_rate'])

    pwmat_mmt = stream_structures(settings['input_file'], format=settings['input_format'], get_forces=True)

    for iepoch in range(settings['epoch']):
        for step, (Rcart, lattice, atom_types, F, Ei) in enumerate(pwmat_mmt):
            start = time.time()
            g, g_dldl, g_dpdl, idxNb = get_dG_dR(Rcart, lattice,
                                                 basis='cosine', M2=M2, M3=M3, Rinner=0, Router=Rc)
            g, g_dldl, g_dpdl = np2torch(g, g_dldl, g_dpdl, dtype=dtype)
            feat_scaled = gscalar[0] * g + gscalar[1]

            nn_out2, d_nn_out2 = d_mlnn(feat_scaled, weights, biases, activation="sigmoid")
            forces = get_forces(g_dldl, g_dpdl, d_nn_out2, idxNb, gscalar[0])
            z = (nn_out2 - escalar[1]) / escalar[0]
            dz = forces / escalar[0]

            y, dy = np2torch(Ei[:, None], F, dtype=dtype)
            F_mse = torch.mean((dz - dy) ** 2)
            E_mse = torch.mean((z - y) ** 2)

            loss = E_mse + F_mse
            mlnn_optimize(loss, optimizer)
            optimizer.step()

            print(step, *torch2np(torch.sqrt(E_mse), torch.sqrt(F_mse)), time.time() - start)

    return list(torch2np(*weights)), list(torch2np(*biases))

