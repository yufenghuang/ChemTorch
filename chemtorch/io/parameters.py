import numpy as np
import pandas as pd
import json


def save_settings(settings):
    with open("settings.json", "w") as f:
        json.dump(settings, f)


def load_settings():
    with open("settings.json", "r") as f:
        settings = json.load(f)
    return settings


def save_weights_biases(weights, biases):
    for i in range(len(weights)):
        w_file = "weights" + str(i) + ".csv"
        b_file = "biases" + str(i) + ".csv"

        w_out = pd.DataFrame(weights[i])
        b_out = pd.DataFrame(biases[i])

        w_out.to_csv(w_file, header=False, index=False, mode="w")
        b_out.to_csv(b_file, header=False, index=False, mode="w")


def load_weights_biases(num):
    weights = []
    biases = []
    for i in range(num):
        w_file = "weights" + str(i) + ".csv"
        b_file = "biases" + str(i) + ".csv"

        w_in = pd.read_csv(w_file, header=None)
        b_in = pd.read_csv(b_file, header=None)

        weights.append(w_in.values)
        biases.append(b_in.values)

    return weights, biases


def save_scalar(feat_a, feat_b, engy_a, engy_b):
    feat_a = feat_a.reshape(-1)
    feat_b = feat_b.reshape(-1)
    engy_a = engy_a.reshape(-1)
    engy_b = engy_b.reshape(-1)

    feat_scalar = pd.DataFrame(np.array([feat_a, feat_b]))
    engy_scalar = pd.DataFrame(np.array([engy_a, engy_b]))

    feat_scalar.to_csv("features_scalar.csv", header=False, index=False, mode="w")
    engy_scalar.to_csv("energies_scalar.csv", header=False, index=False, mode="w")


def load_scalar():
    feat_scalar = pd.read_csv("features_scalar.csv", header=None).values
    engy_scalar = pd.read_csv("energies_scalar.csv", header=None).values

    feat_a = feat_scalar[0].reshape((1, -1))
    feat_b = feat_scalar[1].reshape((1, -1))
    engy_a = engy_scalar[0].reshape((1, -1))
    engy_b = engy_scalar[1].reshape((1, -1))

    return feat_a, feat_b, engy_a, engy_b



