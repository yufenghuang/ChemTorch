def test_read_PWMat():
    from chemtorch.io.read import read_PWMat_movement
    from sys import platform
    filename = "tests/data/MOVEMENT_test"
    if platform == "win32":
        filename = "tests\data\MOVEMENT_test"

    mmt = read_PWMat_movement(filename, get_forces=True, get_velocities=True, get_Ei=True, get_Epot=True)
    n_atoms, lattice, atom_types, R, F, V, Ei, Epot = next(mmt)
    n_atoms, lattice, atom_types, R, F, V, Ei, Epot = next(mmt)
    print("Number of atoms is ", n_atoms)
    print("lattice is ", lattice)
    print("Atom types are ", atom_types)
    print("Atomic position for atom 10 ", R[9])
    print("Atomic forces for atom 10 ", F[9])
    print("Atomic velocities for atom 10 ", V[9])
    print("Atomic energies are ", Ei)
    print("Total potential energy is ", Epot)