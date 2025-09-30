import numpy as np
import blackbox as bb
from scipy.optimize import fsolve

bplus_h11 = 10.817
bplus_h11_unc = 0.005
bminus_h11 = -47.42
bminus_h11_unc = 0.014
spin_h11 = 0.5
mass_h11 = 1
r_naught = 1.3 #fm
r_naught_unc = 0.1 #fm

def b_coherent(params):
    bplus_num = params[0] + 1
    bminus_num = params[0]
    denom = (2*params[0]) + 1
    bplus_coeff = bplus_num/denom
    bminus_coeff = bminus_num/denom
    bplus = bplus_coeff*params[1]
    bminus = bminus_coeff*params[2]
    return bplus + bminus

b_coh_h11 = b_coherent([spin_h11, bplus_h11, bminus_h11])
b_coh_h11_unc = bb.uncertainty_prop(b_coh_h11, [spin_h11, bplus_h11, bminus_h11], [0, bplus_h11_unc, bminus_h11_unc], b_coherent)

print(b_coh_h11)
print(b_coh_h11_unc)

def atomic_radius(params):
    return params[0] * (params[1]**(1/3))

radius_h11 = atomic_radius([r_naught, mass_h11])
radius_h11_unc = bb.uncertainty_prop(radius_h11, [r_naught, mass_h11], [r_naught_unc, 0], atomic_radius)

print(radius_h11)
print(radius_h11_unc)

def ratio(params):
    return params[0]/params[1]

ratio_h11 = ratio([b_coh_h11, radius_h11])
ratio_h11_unc = bb.uncertainty_prop(ratio_h11, [b_coh_h11, radius_h11], [b_coh_h11_unc, radius_h11_unc], ratio)

print(ratio_h11)
print(ratio_h11_unc)

def func(x):
    return 1 - ratio_h11 - (np.tan(x)/x)

def func_upper(x):
    return 1 - (ratio_h11+ratio_h11_unc) - (np.tan(x)/x)

def func_lower(x):
    return 1 - (ratio_h11-ratio_h11_unc) - (np.tan(x)/x)

x_init_guess = 1.5
x_sol = fsolve(func, x_init_guess)
x_upper_sol = fsolve(func_upper, x_init_guess)
x_lower_sol = fsolve(func_lower, x_init_guess)
print(x_sol)
print(x_upper_sol)
print(x_lower_sol)
x_unc = 0.5*np.abs(x_sol[0] - x_upper_sol[0]) + np.abs(x_sol[0] - x_lower_sol[0])
print(x_unc)

def well_depth(params):
    mass_factor = (params[1] + 1) / params[1]
    mass_factor *= 1 / (params[1] ** (2/3))
    mass_factor *= 12.26
    return params[0]**2 * mass_factor

depth_h11 = well_depth([x_sol[0], mass_h11])
depth_unc_h11 = bb.uncertainty_prop(depth_h11, [x_sol[0], mass_h11], [x_unc, 0], well_depth)
print(str(depth_h11) + " +/- " + str(depth_unc_h11))