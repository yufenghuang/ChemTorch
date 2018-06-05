import numpy as np
import torch
import torch.nn
import torch.nn.init
import time
from pydoc import locate


from ..nn import get_weights, get_biases, np2torch, torch2np, d_mlnn, get_forces, mlnn_optimize
from ..ml import get_scalar_csv
from ..io import stream_structures
from ..features import get_dG_dR


def train_force(settings):
    dtype, device = locate(settings['dtype']), settings['device']

    M2, M3 = settings['M2'], settings['M3']
    num_feat, num_engy = M2 + M3 ** 3, 1
    mlp = [num_feat] + settings['hidden_layers'] + [num_engy]
    Rc = settings['Router']
    weights, biases = get_weights(mlp), get_biases(mlp)
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
