metal_to_si = {
    "mass": 1.,  # metal: grams/mole, si: kilograms/mole
    "distance": 1.,  # metal: Angstroms, si: meters
    "time": 1.,  # metal: picoseconds, si: seconds
}

si_to_metal = {k: 1/v for (k,v) in metal_to_si.items()}