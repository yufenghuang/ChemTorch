from .read import read_PWMat_movement
from .. import structure

def stream_structures(filename, format='xyz', **params):
    if format == 'pwmat':
        # convert parameters
        pwmat_pararms = {"get_forces": False, "get_velocities": False, "get_Ei": True, "get_Epot": False}
        for param in params:
            if param in pwmat_pararms:
                pwmat_pararms[param] = bool(params[param])

        mmt = read_PWMat_movement(filename,
                                  get_forces=pwmat_pararms["get_forces"],
                                  get_velocities=pwmat_pararms["get_velocities"],
                                  get_Ei=pwmat_pararms["get_Ei"],
                                  get_Epot=pwmat_pararms["get_Epot"])

        for mmt_out in mmt:
            lattice = mmt_out[1]
            atom_types = mmt_out[2]
            Rfrac = mmt_out[3]

            lattice = structure.standardize_lattice(lattice)
            Rcart = structure.frac2cart(Rfrac, lattice)

            out = [Rcart, lattice, atom_types]

            yield tuple(out) + mmt_out[4:]
