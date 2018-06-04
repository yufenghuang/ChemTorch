import torch
import numpy as np
import torch.nn
from .coord2energy import get_forces


def np2torch(*np_arrays, dtype=torch.float):
    if len(np_arrays) == 1:
        return torch.from_numpy(np_arrays[0]).to(dtype)
    else:
        return tuple([torch.from_numpy(np_array).to(dtype) for np_array in np_arrays])


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
    d_nn_out = torch.matmul(torch.ones_like(node_vals[-1]), weights[-1].t())
    for v, w in zip(reversed(node_vals[1:-1]), reversed(weights[:-1])):
        d_nn_out = torch.matmul(d_nn_out * v * (1 - v), w.t())

    return nn_out, d_nn_out