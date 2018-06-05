from chemtorch.parameters import settings
from chemtorch.training.sannp import train_energy, train_force

settings['epoch'] = 5
weights, biases = train_energy(settings)
weights2, biases2 = train_force(settings, weights=weights, biases=biases)