def test_cart_frac_conversion():
    import numpy as np
    from chemtorch.structure.lattice import standardize_lattice
    from chemtorch.structure.coordinates import frac2cart
    from chemtorch.structure.coordinates import cart2frac

    l0 = []
    l1 = np.random.rand(1, 1) * 5
    l2 = np.random.rand(2, 2) * 5
    l3 = np.random.rand(3, 3) * 5

    l0 = standardize_lattice(l0)
    l1 = standardize_lattice(l1)
    l2 = standardize_lattice(l2)
    l3 = standardize_lattice(l3)


    L = [l0, l1, l2, l3]

    # 3 dimensional coordinates
    R = np.random.rand(10, 3) * 5
    for l in L:
        R0 = R.copy()
        R1 = frac2cart(cart2frac(R0, l), l)
        R0 = R1.copy()
        R1 = frac2cart(cart2frac(R0, l), l)
        print(len(l), "dimension conversional")
        print("First row of the coordinate is", R1[0])
        print("maximum error is", np.max(R1 - R0))

    # 4 dimensional coordinates
    l4 = np.random.rand(4, 4) * 5
    l4 = standardize_lattice(l4)
    L.append(l4)
    R = np.random.rand(10, 4) * 5
    for l in L:
        R0 = R.copy()
        R1 = frac2cart(cart2frac(R0, l), l)
        R0 = R1.copy()
        R1 = frac2cart(cart2frac(R0, l), l)
        print(len(l), "dimension conversional")
        print("First row of the coordinate is", R1[0])
        print("maximum error is", np.max(R1 - R0))


