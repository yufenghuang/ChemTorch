from chemtorch.parameters import settings
from chemtorch.io import stream_structures
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from torch.autograd import grad


def pcosine(Rij, num_basis, start=-1.0, stop=1.0):
    """ Piecewise cosine functions on an array xin, \
        return a matrix of dimension dim(xin)*num_basis"""

    assert Rij.dim() <= 1, "Rij should be a 0- or 1-dimensional array, " \
                              "but the shape of xin is " + str(Rij.shape)
    assert num_basis >= 2, "num_basis has to be greater than 2"
    num_basis = int(num_basis)

    h = (stop - start)/(num_basis-1)
    nodes = torch.linspace(start, stop, num_basis)

    y = Rij[:, None] - nodes
    out = torch.zeros_like(y)

    out[torch.abs(y) < h] = torch.cos(y[torch.abs(y) < h] * math.pi / h)/2 + 1/2

    return out


def dpcosine(Rij, num_basis, start=-1, stop=1):
    """ Derivative of the piecewise cosine functions on an array xin, \
        return a matrix of dimension dim(xin)*num_basis"""

    assert Rij.dim() <= 1, "Rij should be a 0- or 1-dimensional array, " \
                              "but the shape of xin is " + str(Rij.shape)
    assert num_basis >= 2, "num_basis has to be greater than 2"
    num_basis = int(num_basis)

    h = (stop - start)/(num_basis-1)
    nodes = torch.linspace(start, stop, num_basis)

    y = Rij[:, None] - nodes
    out = torch.zeros_like(y)

    out[torch.abs(y) < h] = -1/2 * math.pi / h * torch.sin(y[torch.abs(y) < h] * math.pi / h)

    return out


pwmat_mmt = stream_structures(settings['input_file'], format=settings['input_format'], get_forces=True)
Rcart, lattice, atom_types, F, Ei = next(pwmat_mmt)


xR = torch.linspace(-2, 2, 1000)
yR = pcosine(xR, 10)

plt.plot(xR.numpy(), yR.numpy())

Rt = torch.from_numpy(Rcart)
Rt = Rt.to(torch.float)
Rt.requires_grad = True

phi = pcosine(Rt[Rt>0], 10, start=5., stop=float(torch.max(Rt)-5))
dphi = torch.zeros_like(Rt)
dphi[Rt>0] = torch.sum(dpcosine(Rt[Rt>0], 10, start=5., stop=float(torch.max(Rt)-5)), dim=1)

plt.figure()
plt.plot(Rt[Rt>0].detach().numpy(), phi.detach().numpy(), '.')

dphi2 = grad(torch.sum(phi), Rt, create_graph=True)

plt.figure()
plt.plot(Rt[Rt>0].detach().numpy(), dphi[Rt>0].detach().numpy(),'o')
plt.plot(Rt[Rt>0].detach().numpy(), (dphi2[0])[Rt>0].detach().numpy(),'^')

# Test gradient
x1 = torch.randn(1, requires_grad=True)
print(x1)
y = x1 ** 3
dy = grad(y, x1, create_graph=True)
dy[0].backward()
print(x1, dy, x1.grad)