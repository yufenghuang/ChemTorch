from sys import platform


settings = {

    "dtype": 'torch.float',
    "device": 'cpu',

    "learning_rate": 1e-4,
    "activation": "sigmoid",
    "hidden_layers": [50, 50],
    "epoch": 1,
    "chunk_size": 1000,

    'basis': 'cosine',
    'M2': 25,
    'M3': 5,
    'Rinner': 0.0,
    'Router': 6.0,

    'input_file': "tests\data\MOVEMENT.train" if platform == 'win32' else "tests/data/MOVEMENT.train",
    'input_format': "pwmat",

    "valid_feat_file": "tests\\data\\vfeat" if platform == 'win32' else "tests/data/vfeat",
    "valid_engy_file": "tests\\data\\vengy" if platform == 'win32' else "tests/data/vengy",
    "test_feat_file": "tests\\data\\tfeat" if platform == 'win32' else "tests/data/tfeat",
    "test_engy_file": "tests\\data\\tengy" if platform == 'win32' else "tests/data/tengy",
    "train_feat_file": "tests\\data\\feat" if platform == 'win32' else "tests/data/feat",
    "train_engy_file": "tests\\data\\engy" if platform == 'win32' else "tests/data/engy",

}


parameters = {

}