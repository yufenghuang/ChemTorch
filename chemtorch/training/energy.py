import torch
from pydoc import locate
import pandas as pd
import time

from ..nn import get_weights, get_biases, np2torch, torch2np, mlnn, mlnn_optimize
from ..ml import get_scalar_csv


def train_energy(settings):
    dtype, device = locate(settings['dtype']), settings['device']

    M2, M3 = settings['M2'], settings['M3']
    num_feat, num_engy = M2 + M3 ** 3, 1
    mlp = [num_feat] + settings['hidden_layers'] + [num_engy]
    weights, biases = get_weights(mlp), get_biases(mlp)
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
