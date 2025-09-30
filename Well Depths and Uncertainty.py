import numpy as np
import blackbox as bb
from scipy.optimize import fsolve

spin, bplus, bplus_unc, bminus, bminus_unc, n, mass_num = np.genfromtxt('Neutron-Spins-_-Scattering-Lengths-Sheet1-_1_.txt', unpack=True)

elements = ["1H1", "1H2", "1H3", "2He3", "2He4", "3Li6", "3Li7", "4Be9", "5B10", "5B11", "6C12", "6C13", "7N14", "7N15"]

r_naught = 1.3 #fm
r_naught_unc = 0.1 #fm

def b_coherent(params):
    bplus_num = params[0] + 1
    bminus_num = params[0]
    denom = (2*params[0]) + 1
    bplus_coeff = bplus_num/denom
    bminus_coeff = bminus_num/denom
    bplus_c = bplus_coeff*params[1]
    bminus_c = bminus_coeff*params[2]
    return bplus_c + bminus_c

def atomic_radius(params):
    return params[0] * (params[1]**(1/3))

def ratio(params):
    return params[0]/params[1]

def well_depth(params):
    mass_factor = (params[1] + 1) / params[1]
    mass_factor *= 1 / (params[1] ** (2/3))
    mass_factor *= 12.26
    return params[0]**2 * mass_factor

x_init_guesses = [1.5, 1.7, 4.8]

for i in range(len(spin)):
    if bminus[i] == 0:
        b_coh = bplus[i]
        b_coh_unc = bplus_unc[i]
    else:
        b_coh = b_coherent([spin[i], bplus[i], bminus[i]])
        b_coh_unc = bb.uncertainty_prop(b_coh, [spin[i], bplus[i], bminus[i]], [0, bplus_unc[i], bminus_unc[i]], b_coherent)
    radius = atomic_radius([r_naught, mass_num[i]])
    radius_unc = bb.uncertainty_prop(radius, [r_naught, mass_num[i]], [r_naught_unc, 0], atomic_radius)
    coh_ratio = ratio([b_coh, radius])
    coh_ratio_unc = bb.uncertainty_prop(coh_ratio, [b_coh, radius], [b_coh_unc, radius_unc], ratio)

    def func(x):
        return 1 - coh_ratio - np.tan(x)/x

    def func_upper(x):
        return 1 - (coh_ratio + coh_ratio_unc) - np.tan(x)/x

    def func_lower(x):
        return 1 - (coh_ratio - coh_ratio_unc) - np.tan(x)/x

    x_guess = x_init_guesses[int(n[i])]
    x_sol = fsolve(func, x_guess)[0]
    x_sol_upper = fsolve(func_upper, x_guess)[0]
    x_sol_lower = fsolve(func_lower, x_guess)[0]
    x_unc = 0.5*(np.abs(x_sol - x_sol_upper) + np.abs(x_sol - x_sol_lower))

    depth = well_depth([x_sol, mass_num[i]])
    depth_unc = bb.uncertainty_prop(depth, [x_sol, mass_num[i]], [x_unc, 0], well_depth)

    element = elements[i]
    print("Well Depth for " + element + " : " + str(depth) + " +/- " + str(depth_unc))