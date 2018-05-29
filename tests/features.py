def test_p_cos():
    from chemtorch.features.basis.piecewise_cosine import piecewise_cosine
    import matplotlib.pyplot as plt
    import numpy as np
    n_basis = 5
    x = np.linspace(0, 8, 1000)
    y = piecewise_cosine(x, n_basis, 2, 6.2-(6.2-2)/n_basis)
    plt.figure()
    plt.plot(x, y)


def test_diff_p_cos():
    from chemtorch.features.basis.piecewise_cosine import diff_p_cosine
    import matplotlib.pyplot as plt
    import numpy as np
    n_basis = 5
    x = np.linspace(0, 8, 1000)
    y = diff_p_cosine(x, n_basis, 2, 6.2 - (6.2 - 2) / n_basis)
    plt.figure()
    plt.plot(x, y)

def get_g1_old(Ri, eta, Rs, Rc):
    import numpy as np

    def fc(Rij, Rc):
        Rij_ = np.array(Rij)
        R_ = np.zeros(Rij_.shape)
        R_[Rij_ < Rc] = 0.5 * (np.cos(Rij_[Rij_ < Rc] * np.pi / Rc) + 1)
        return R_

    def g1(Rij, Rs_, eta_, Rc):
        return np.exp(-eta_ * (Rij - Rs_) ** 2) * fc(Rij, Rc)

    feats = np.zeros((len(Ri), eta.size * Rs.size))
    for i in range(Rs.size):
        for j in range(eta.size):
            ifeat = i * eta.size + j
            G1 = np.zeros(Ri.shape)
            G1[Ri > 0.] = g1(Ri[Ri > 0.], Rs[i], eta[j], Rc)
            feats[:, ifeat] = G1.sum(axis=1)

    return feats

def get_g2_old(ll, eta, zeta, Rc, Di, Dj, Dc):
    import numpy as np

    def fc(Rij, Rc):
        Rij_ = np.array(Rij)
        R_ = np.zeros(Rij_.shape)
        R_[Rij_ < Rc] = 0.5 * (np.cos(Rij_[Rij_ < Rc] * np.pi / Rc) + 1)
        return R_

    def g2(Rij, Rik, Rjk, eta_, zeta_, ll_, Rc):  ##ll: lambda
        cos = (Rij ** 2 + Rik ** 2 - Rjk ** 2) / (2 * Rij * Rik)
        return 2. ** (1 - zeta_) * (1 + ll_ * cos) ** zeta_ * np.exp(-eta_ * (Rij ** 2 + Rik ** 2 + Rjk ** 2)) \
               * fc(Rij, Rc) * fc(Rik, Rc) * fc(Rjk, Rc)

    feats = np.zeros((len(Di), ll.size * zeta.size * eta.size))
    for i in range(ll.size):
        for j in range(eta.size):
            for k in range(zeta.size):
                ifeat = i * eta.size * zeta.size + j * zeta.size + k
                G2 = np.zeros(Dc.shape)
                G2[Dc > 0.] = g2(Di[Dc > 0.], Dj[Dc > 0.], Dc[Dc > 0.], eta[j], zeta[k], ll[i], Rc)
                feats[:, ifeat] = G2.sum(axis=2).sum(axis=1)

    return feats


def test_gaussian_g1():
    from chemtorch.features.basis.gaussian_like import g1
    import numpy as np
    Rc = 6.2
    eta = np.arange(1, 6) ** 2 / Rc ** 2
    Rs = np.arange(1, 11) * Rc / 10
    Rij = np.random.rand(30, 30) * 8
    Rij[Rij < 2] = 0
    G1 = np.zeros((*Rij.shape, len(eta), len(Rs)))
    G1[Rij>0] = g1(Rij[Rij > 0], eta, Rs, Rc)
    G1_new = G1.sum(axis=1).transpose([0,2,1]).reshape((-1, len(eta)*len(Rs)))
    G1_old = get_g1_old(Rij, eta, Rs, Rc)
    print("maximum error is ", np.max(np.abs(G1_new - G1_old)))


def test_gaussian_g2():
    from chemtorch.features.basis.gaussian_like import g2
    import numpy as np
    Rc = 6.2
    ll = np.array([-1., 1])
    eta = np.arange(1, 6) ** 2 / Rc ** 2
    zeta = np.arange(0, 6)

    Di = np.random.rand(50,30,30) * 8
    Di[Di<2] = 0
    Dj = np.transpose(Di, [0,2,1])
    Di[Dj == 0] = 0
    Dj[Di == 0] = 0
    theta = np.random.rand(*Di.shape) * np.pi
    Dc = np.sqrt(Di ** 2 + Dj ** 2 - 2 * Di * Dj * np.cos(theta))
    Dc[Di == 0] = 0

    G2 = np.zeros((*Dc.shape, len(zeta), len(eta), len(ll)))
    G2[Dc > 0] = g2(Di[Dc > 0], Dj[Dc > 0], Dc[Dc > 0], zeta, eta, ll, Rc)
    G2_old = get_g2_old(ll, eta, zeta, Rc, Di, Dj, Dc)
    G2_new = G2.sum(axis=1).sum(axis=1).transpose([0,3,2,1]).reshape((len(Di), -1))
    print("maximum error is ", np.max(np.abs(G2_new - G2_old)))