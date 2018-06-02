# Test run #1

import numpy as np
import torch
import torch.nn
import torch.nn.init
from chemtorch.structure.coordinates import frac2cart, cart2frac
from chemtorch.structure.lattice import standardize_lattice

dtype = torch.float
device = torch.device("cpu")

R = np.random.rand(10,3)
lattice = standardize_lattice(np.random.rand(3,3)*5)
R0 = frac2cart(R, lattice)

feat = R0.copy()

num_feat = 60
num_engy = 1
mlp = [num_feat, 50, 50, 50, num_engy]
weights = [torch.zeros([mlp[i], mlp[i+1]], dtype=dtype, device=device, requires_grad=True) for i in range(len(mlp)-1)]
biases = [torch.zeros([1, mlp[i+1]], dtype=dtype, device=device, requires_grad=True) for i in range(len(mlp)-1)]
optimizer = torch.optim.Adam(biases + weights, lr=1e-4)

with torch.no_grad():
    for w in weights:
        w = torch.nn.init.xavier_normal_(w)

input = torch.randn(100, num_feat, dtype=dtype, device=device)
output = torch.randn(100, 1, dtype=dtype, device=device)

# for i in range(len(weights)):
#     input = torch.matmul(input, weights[i])+biases[i]
sigmoid = torch.nn.Sigmoid()
for w,b in zip(weights, biases):
    input = sigmoid(torch.matmul(input, w) + b)

L = torch.sum((input - output)**2)
L.backward()
optimizer.step()
# for i in range(len(weights)):
#     weights[i].grad.zero_()
# for i in range(len(biases)):
#     biases[i].grad.zero_()
for w in weights:
    w.grad.zero_()
for b in biases:
    b.grad.zero_()