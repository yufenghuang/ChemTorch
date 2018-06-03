import torch
import torch.nn


def get_weights(mlp, dtype=torch.float, device=torch.device('cpu'), xavier=False):
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
        print("Activation function", activation, "not recognizable")
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


