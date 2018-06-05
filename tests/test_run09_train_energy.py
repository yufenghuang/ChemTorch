from chemtorch.parameters import settings
from chemtorch.training.energy import train_energy
from chemtorch.training.force import train_force

train_energy(settings)
train_force(settings)