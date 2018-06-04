import torch
import numpy as np

def get_forces(dGl_dRl, dGp_dRl, d_nn_out2, idxNb, feat_a):

    n_atoms, maxNb = idxNb.shape
    dim = dGl_dRl.shape[-1]

    d_nn_out2 = d_nn_out2 * feat_a
    fll = torch.matmul(d_nn_out2[:, None, :], dGl_dRl).squeeze()

    fln = torch.zeros((n_atoms, maxNb, dim))
    fln[np.where(idxNb > 0)] = torch.matmul(d_nn_out2[idxNb[idxNb > 0] - 1][:, None, :],
                                            dGp_dRl[np.where(idxNb > 0)]).squeeze()
    forces = -torch.sum(fln, dim=1) - fll

    return forces

