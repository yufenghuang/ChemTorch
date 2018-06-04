import numpy as np


def piecewise_cosine(Rij, num_basis, start=-1, stop=1):
    """ Piecewise cosine functions on an array xin, \
        return a matrix of dimension dim(xin)*num_basis"""

    assert np.ndim(Rij) <= 1, "Rij should be a 0- or 1-dimensional array, " \
                              "but the shape of xin is " + str(Rij.shape)
    assert num_basis >= 2, "num_basis has to be greater than 2"
    num_basis = int(num_basis)

    h = (stop - start)/(num_basis-1)
    nodes = np.linspace(start, stop, num_basis)

    y = Rij[:, None] - nodes
    out = np.zeros_like(y)

    out[np.abs(y) < h] = np.cos(y[np.abs(y) < h] * np.pi / h)/2 + 1/2

    return out


def diff_p_cosine(Rij, num_basis, start=-1, stop=1):
    """ Derivative of the piecewise cosine functions on an array xin, \
        return a matrix of dimension dim(xin)*num_basis"""

    assert np.ndim(Rij) <= 1, "Rij should be a 0- or 1-dimensional array, " \
                              "but the shape of xin is " + str(Rij.shape)
    assert num_basis >= 2, "num_basis has to be greater than 2"
    num_basis = int(num_basis)

    h = (stop - start)/(num_basis-1)
    nodes = np.linspace(start, stop, num_basis)

    y = Rij[:, None] - nodes
    out = np.zeros_like(y)

    out[np.abs(y) < h] = -1/2 * np.pi / h * np.sin(y[np.abs(y) < h] * np.pi / h)

    return out

def get_g1():
    pass


def get_g2():
    pass


def get_d_g1():
    pass


def get_d_g2():
    pass


def get_features(Rij, Rijk, M2, M3, Rinner=0, Router=6):

    pass


def get_d_features(Rij, Rijk, Rhat, M2, M3, Rinner=0, Router=6):
    nAtoms = len(Rij)
    maxNb = Rij.shape[1]
    dim = Rhat.shape[-1]

    phi2 = np.zeros((*Rij.shape, M2))
    phi3_ij = np.zeros((*Rij.shape, M3))
    phi3_ijk = np.zeros((*Rijk.shape, M3))
    d_phi2 = np.zeros((*Rij.shape, M2))
    d_phi3 = np.zeros((*Rij.shape, M3))

    R2 = Router - (Router - Rinner) / M2
    R3 = Router - (Router - Rinner) / M3

    phi2[Rij > 0] = piecewise_cosine(Rij[Rij > 0], M2, start=Rinner, stop=R2)
    phi3_ij[Rij > 0] = piecewise_cosine(Rij[Rij > 0], M3, start=Rinner, stop=R3)
    phi3_ijk[Rijk > 0] = piecewise_cosine(Rijk[Rijk>0], M3, start=Rinner, stop=R3)
    d_phi2[Rij > 0] = diff_p_cosine(Rij[Rij>0], M2, start=Rinner, stop=R2)
    d_phi3[Rij > 0] = diff_p_cosine(Rij[Rij>0], M3, start=Rinner, stop=R3)

    z2b = d_phi2[:, :, :, None] * -Rhat[:, :, None, :]
    z3a = np.matmul(phi3_ij.transpose([0, 2, 1]), phi3_ijk.reshape((nAtoms, maxNb, -1)))
    z3a = z3a.reshape((nAtoms, M3, maxNb, M3)).transpose([0, 2, 1, 3])
    z3b = d_phi3[:, :, :, None] * -Rhat[:, :, None, :]

    # features
    g2 = phi2.sum(axis=1)
    g3 = np.matmul(phi3_ij.transpose([0, 2, 1]), z3a.reshape((nAtoms, maxNb, -1))).reshape((nAtoms, -1))
    g = np.concatenate([g2, g3], axis=1)

    # derivatives
    # 2-body:
    g2_dldl = z2b.sum(axis=1)
    g2_dpdl = z2b

    # 3-body:
    g3_dldl = np.matmul(z3b.reshape((nAtoms, maxNb, -1)).transpose([0, 2, 1]), z3a.reshape((nAtoms, maxNb, -1)))
    g3_dldl = g3_dldl.reshape((nAtoms, M3, dim, M3, M3)).transpose([0, 1, 3, 4, 2])
    g3_dldl = g3_dldl + g3_dldl.transpose([0, 2, 1, 3, 4])

    g3_dpdl_A = z3a.transpose([0, 1, 3, 2])[:, :, None, :, :, None] * z3b[:, :, :, None, None, :]
    g3_dpdl_C = np.matmul(z3b.reshape((nAtoms, maxNb, -1)).transpose([0, 2, 1]),
                          phi3_ijk.reshape(nAtoms, maxNb, -1))
    g3_dpdl_C = g3_dpdl_C.reshape((nAtoms, M3, dim, maxNb, M3)).transpose([0, 3, 4, 1, 2])
    g3_dpdl_C = g3_dpdl_C[:, :, None, :, :, :] * phi3_ij[:, :, :, None, None, None]

    g3_dpdl = g3_dpdl_A + g3_dpdl_C
    g3_dpdl = g3_dpdl + g3_dpdl.transpose([0, 1, 3, 2, 4, 5])

    g_dldl = np.concatenate([g2_dldl, g3_dldl.reshape((nAtoms, -1, dim))], axis=1)
    g_dpdl = np.concatenate([g2_dpdl, g3_dpdl.reshape((nAtoms, maxNb, -1, dim))], axis=2)

    return g, g_dldl, g_dpdl


