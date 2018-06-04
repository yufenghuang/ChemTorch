import numpy as np


def read_PWMat_movement(filename, get_forces=False, get_velocities=False, get_Ei=False, get_Epot=False):
    with open(filename, 'r') as file:
        line = file.readline()
        while "Iteration" in line:
            n_atoms = int(line.split()[0])
            Epot = float(line.split()[5])
            lattice = np.zeros((3,3))
            R = np.zeros((n_atoms, 3))
            forces = np.zeros((n_atoms, 3))
            velocities = np.zeros((n_atoms, 3))
            energies = np.zeros((n_atoms))
            atom_types = np.zeros((n_atoms), dtype=int)

            while True:
                line = file.readline()
                if "Lattice" in line:
                    lattice[0, :] = np.array(file.readline().split(), dtype=float)
                    lattice[1, :] = np.array(file.readline().split(), dtype=float)
                    lattice[2, :] = np.array(file.readline().split(), dtype=float)
                elif "Position" in line:
                    for i in range(n_atoms):
                        line = file.readline().split()
                        atom_types[i] = int(line[0])
                        R[i, :] = np.array(line[1:4])
                    R = R - np.floor(R)
                elif "Force" in line:
                    for i in range(n_atoms):
                        forces[i, :] = np.array(file.readline().split()[1:4])
                elif "Velocity" in line:
                    for i in range(n_atoms):
                        velocities[i, :] = np.array(file.readline().split()[1:4])
                elif "Atomic-Energy" in line:
                    for i in range(n_atoms):
                        energies[i] = float(file.readline().split()[1])
                else:
                    break

            # returning results
            out = [n_atoms, lattice, atom_types, R]
            if get_forces:
                out.append(-forces)
            if get_velocities:
                out.append(velocities)
            if get_Ei:
                out.append(energies)
            if get_Epot:
                out.append(Epot)
            yield tuple(out)

            # prepare for reading the next structure
            line = file.readline()


def read_outcar(filename, forces=False, velocities=False):
    pass


def read_xyz(filename):
    pass


def read_special_xyz(filename, total_energy=False, forces=False, velocities=False):
    pass


