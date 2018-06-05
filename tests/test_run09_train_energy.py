from chemtorch.parameters import settings
from chemtorch.training.sannp import train_energy, train_force
from chemtorch.ml import get_scalar_csv

gscalar = get_scalar_csv(settings["valid_feat_file"])
escalar = get_scalar_csv(settings["valid_engy_file"])

settings['epoch'] = 1
weights, biases = train_energy(settings, gscalar=gscalar, escalar=escalar)
weights, biases = train_force(settings, weights=weights, biases=biases, gscalar=gscalar, escalar=escalar)