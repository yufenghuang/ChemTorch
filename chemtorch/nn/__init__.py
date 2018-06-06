import torch
import numpy as np
import torch.nn
from .coord2energy import get_forces
from .coord2energy import get_distances, get_nb, pcosine
from ..ml import get_scalar_csv


def np2torch(*np_arrays, dtype=torch.float):
    if len(np_arrays) == 1:
        x = np_arrays[0]
        return torch.from_numpy(np.array(x)).to(dtype)
    else:
        return tuple([torch.from_numpy(np.array(x)).to(dtype) for x in np_arrays])


def torch2np(*torch_arrays, dtype=np.float64):
    if len(torch_arrays) == 1:
        return torch_arrays[0].detach().numpy().astype(dtype)
    else:
        return tuple([torch_array.detach().numpy().astype(dtype) for torch_array in torch_arrays])


def get_weights(mlp, dtype=torch.float, device=torch.device('cpu'), xavier=True):
    weights = [torch.zeros([mlp[i], mlp[i + 1]], dtype=dtype, device=device, requires_grad=True)
               for i in range(len(mlp) - 1)]
    if xavier:
        with torch.no_grad():
            for w in weights:
                torch.nn.init.xavier_normal_(w)
    return weights


def get_biases(mlp, dtype=torch.float, device=torch.device('cpu')):
    return [torch.zeros([1, mlp[i+1]], dtype=dtype, device=device, requires_grad=True) for i in range(len(mlp)-1)]


def mlnn(nn_in, weights, biases=None, activation="sigmoid"):
    act_fn = torch.nn.Sigmoid()
    if activation == "sigmoid":
        pass
    else:
        print("Activation function", activation, "is not recognizable")
        print("Sigmoid function will be used instead.")
    pass

    if biases is None:
        for w in weights:
            nn_in = act_fn(torch.matmul(nn_in, w))
    else:
        for w, b in zip(weights, biases):
            nn_in = act_fn(torch.matmul(nn_in, w) + b)

    return nn_in


def mlnn_optimize(loss, optimizer):
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def d_mlnn(nn_in, weights, biases, activation='sigmoid'):
    act_fn = torch.nn.Sigmoid()
    if activation == "sigmoid":
        pass
    else:
        print("Activation function", activation, "not recognizable")
        print("Sigmoid function will be used instead.")
    pass

    node_vals = [nn_in]

    # forward pass to obtain the node values at each layer
    if biases is None:
        for w, b in zip(weights, biases):
            node_vals.append(act_fn(torch.matmul(node_vals[-1], w)))
    else:
        for w, b in zip(weights, biases):
            node_vals.append(act_fn(torch.matmul(node_vals[-1], w) + b))

    nn_out = node_vals[-1]

    # back propagate to obtain the derivative
    # d_nn_out = torch.matmul(torch.ones_like(node_vals[-1]), weights[-1].t())
    d_nn_out = 1
    for v, w in zip(reversed(node_vals[1:]), reversed(weights)):
        d_nn_out = torch.matmul(d_nn_out * v * (1 - v), w.t())

    return nn_out, d_nn_out


def R2G(Rcart, lattice, M2=25, M3=5, Rinner=0.0, Router=6.0):

    R2 = Router - (Router - Rinner) / M2
    R3 = Router - (Router - Rinner) / M3

    nblist, Rij = get_nb(Rcart, lattice, Router)
    dij, dijk, Rhat = get_distances(Rij)

    phi2b = torch.zeros(*dij.shape, M2)
    phi3b_ij = torch.zeros(*dij.shape, M3)
    phi3b_ijk = torch.zeros(*dijk.shape, M3)

    phi2b[dij > 0] = pcosine(dij[dij > 0], M2, start=Rinner, stop=R2)
    phi3b_ij[dij > 0] = pcosine(dij[dij > 0], M3, start=Rinner, stop=R3)
    phi3b_ijk[dijk > 0] = pcosine(dijk[dijk > 0], M3, start=Rinner, stop=R3)


    G2 = torch.sum(phi2b, dim=1)

    # shape of G3 will be: natoms x alpha x gamma x maxNb
    G3 = torch.matmul(phi3b_ij.transpose(1,2)[:, :, None, None, :],
                      phi3b_ijk.transpose(1,3)[:, None, :, :, :]).squeeze()

    # shape of G3 will be: natoms x alpha x beta x gamma
    G3 = torch.matmul(phi3b_ij.transpose(1,2)[:, None, :, None, :],
                      G3.transpose(2,3)[:, :, None, :, :]).squeeze()

    G = torch.cat((G2, G3.reshape(len(G3), -1)), 1)
    return G


def G2E(G, weights, biases, gscalar, escalar):
    G = gscalar[0] * G + gscalar[1]

    E = mlnn(G, weights, biases)

    return E

def R2E(Rcart, lattice, weights, biases, gscalar, escalar, settings):
    settings['Router'] = 6.2
    M2 = settings['M2']
    M3 = settings['M3']
    Rinner = settings['Rinner']
    Router = settings['Router']
    R2 = Router - (Router - Rinner) / M2
    R3 = Router - (Router - Rinner) / M3

    nblist, Rij = get_nb(Rcart, lattice, settings['Router'])
    dij, dijk, Rhat = get_distances(Rij)

    phi2b = torch.zeros(*dij.shape, settings['M2'])
    phi3b_ij = torch.zeros(*dij.shape, settings['M3'])
    phi3b_ijk = torch.zeros(*dijk.shape, settings['M3'])


    phi2b[dij > 0] = pcosine(dij[dij > 0], settings['M2'], start=settings['Rinner'], stop=R2)
    phi3b_ij[dij > 0] = pcosine(dij[dij > 0], settings['M3'], start=settings['Rinner'], stop=R3)
    phi3b_ijk[dijk > 0] = pcosine(dijk[dijk > 0], settings['M3'], start=settings['Rinner'], stop=R3)


    G2 = torch.sum(phi2b, dim=1)

    # shape of G3 will be: natoms x alpha x gamma x maxNb
    G3 = torch.matmul(phi3b_ij.transpose(1,2)[:, :, None, None, :],
                      phi3b_ijk.transpose(1,3)[:, None, :, :, :]).squeeze()

    # shape of G3 will be: natoms x alpha x beta x gamma
    G3 = torch.matmul(phi3b_ij.transpose(1,2)[:, None, :, None, :],
                      G3.transpose(2,3)[:, :, None, :, :]).squeeze()

    G = torch.cat((G2, G3.reshape(len(G3), -1)), 1)

    G = gscalar[0] * G + gscalar[1]

    E = mlnn(G, weights, biases)

    return E