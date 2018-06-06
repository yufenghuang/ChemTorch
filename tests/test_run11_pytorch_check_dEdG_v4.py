from chemtorch.parameters import settings
from chemtorch.training.sannp import train_energy, train_force
from chemtorch.ml import get_scalar_csv
from pydoc import locate
import chemtorch.nn as ct_nn
import chemtorch.io as ct_io

from chemtorch.io.parameters import load_weights_biases

from torch.autograd import grad
import time
import torch

settings["Router"] = 6.2

dtype, device = locate(settings['dtype']), settings['device']

weights, biases = load_weights_biases(len(settings['hidden_layers'])+1)
weights = list(ct_nn.np2torch(*weights))
biases = list(ct_nn.np2torch(*biases))
# for (w, b) in zip(weights, biases): w.requires_grad, b.requires_grad = True, True

gscalar = get_scalar_csv(settings["valid_feat_file"])
escalar = get_scalar_csv(settings["valid_engy_file"])

gscalar, escalar = list(ct_nn.np2torch(*gscalar, dtype=dtype)), list(ct_nn.np2torch(*escalar, dtype=dtype))

M2, M3 = settings['M2'], settings['M3']
num_feat, num_engy = M2 + M3**3, 1
mlp = [num_feat] + settings['hidden_layers'] + [num_engy]
Rc = settings['Router']

# optimizer = torch.optim.Adam(biases + weights, lr=settings['learning_rate'])

filename = settings['input_file']
pwmat_mmt = ct_io.stream_structures(filename, format=settings['input_format'], get_forces=True)
Rcart, lattice, atom_types, Fi, Ei = next(pwmat_mmt)

Ei = Ei-272
start = time.time()
Rcart, lattice, Fi, Ei = ct_nn.np2torch(Rcart, lattice, Fi, Ei, dtype=dtype)
# Rcart.requires_grad = True

G = ct_nn.R2G(Rcart, lattice, M2=M2, M3=M3, Rinner=settings["Rinner"], Router=settings['Router'])

Gscaled = G * gscalar[0] + gscalar[1]

Gscaled.requires_grad=True

Ep = ct_nn.mlnn(Gscaled, weights, biases).squeeze()

torch.sum(Ep).backward()

dEdG = Gscaled.grad

i, j  = 0, 10
for i in range(10, 20):
    Gscaled[i, j] = Gscaled[i, j] + 0.001
    Ep2 = ct_nn.mlnn(Gscaled, weights, biases).squeeze()
    print("==================")
    print(dEdG[i, j])
    print((Ep2[i] - Ep[i])/0.001)
    print((torch.sum(Ep2) - torch.sum(Ep))/0.001)


# dEdG = -grad(torch.sum(Ep), Gscaled, create_graph=True)[0]

# Fp = -grad(torch.sum(Ep), Rcart, create_graph=True)[0]

