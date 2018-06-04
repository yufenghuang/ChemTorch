import torch


settings = {
    "dtype": torch.float,
    "device": torch.device('cpu'),
}


nn_params = {
    "activation": "sigmoid",
    "hidden_layers": [50, 50],
}


basis_params = {
    'basis': 'cosine',
    'M2': 25,
    'M3': 5,
    'Rinner': 0.0,
    'Router': 6,
}

