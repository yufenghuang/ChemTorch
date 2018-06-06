from chemtorch.parameters import settings
from chemtorch.training.sannp import train_energy, train_force
from chemtorch.ml import get_scalar_csv
from pydoc import locate
import chemtorch.nn as ct_nn
import chemtorch.io as ct_io

from torch.autograd import grad
import time
import torch

settings["Router"] = 6.2

dtype, device = locate(settings['dtype']), settings['device']


gscalar = get_scalar_csv(settings["valid_feat_file"])
escalar = get_scalar_csv(settings["valid_engy_file"])

gscalar, escalar = list(ct_nn.np2torch(*gscalar, dtype=dtype)), list(ct_nn.np2torch(*escalar, dtype=dtype))

M2, M3 = settings['M2'], settings['M3']
num_feat, num_engy = M2 + M3**3, 1
mlp = [num_feat] + settings['hidden_layers'] + [num_engy]
Rc = settings['Router']

weights, biases = ct_nn.get_weights(mlp), ct_nn.get_biases(mlp)
optimizer = torch.optim.Adam(biases + weights, lr=settings['learning_rate'])

for iEpoch in range(15):
    filename = settings['input_file']
    pwmat_mmt = ct_io.stream_structures(filename, format=settings['input_format'], get_forces=True)
    for i, (Rcart, lattice, atom_types, Fi, Ei) in enumerate(pwmat_mmt):
        Ei = Ei-272
        start = time.time()
        Rcart, lattice, Fi, Ei = ct_nn.np2torch(Rcart, lattice, Fi, Ei, dtype=dtype)
        Rcart.requires_grad = True

        G = ct_nn.R2G(Rcart, lattice, M2=M2, M3=M3, Rinner=settings["Rinner"], Router=settings['Router'])

        G = G * gscalar[0] + gscalar[1]

        Ep = ct_nn.mlnn(G, weights, biases).squeeze()

        Fp = -grad(torch.sum(Ep), Rcart, create_graph=True)[0]

        Emse = torch.mean((Ep - Ei)**2)
        Fmse = torch.mean((Fp - Fi)**2)

        loss = Emse + Fmse
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        Rcart.grad.zero_()

        with torch.no_grad():
            print(iEpoch, i, torch.sqrt(Emse).numpy(), torch.sqrt(Fmse).numpy(), time.time()-start)
