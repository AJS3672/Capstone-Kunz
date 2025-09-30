import numpy as np
import pandas as pd
import blackbox as bb

df = pd.read_csv('Neutron Spins & Scattering Lengths - Sheet1 (2).csv', sep=',')

r_naught = 1.3 #fm

def atomic_radius(params):
    return params[0] * (params[1]**(1/3))

def ratio(params):
    return params[0]/params[1]

def well_depth(params):
    mass_factor = (params[1] + 1) / params[1]
    mass_factor *= 1 / (params[1] ** (2/3))
    mass_factor *= 12.26
    return params[0]**2 * mass_factor

#X = np.linspace(0, 6, 10000)
#Y = np.tan(X)/X

left_hand_values_plus = np.empty(len(df['Isotope']))
left_hand_values_minus = np.empty(len(df['Isotope']))

for i in range(len(df['Isotope'])):
    radius = atomic_radius([r_naught, df['A'][i]])
    for j in range(2):
        if j == 0:
            b = df['b+'][i]
        else:
            b = df['b-'][i]
        value = 1 - ratio([b, radius])
        if j == 0:
            left_hand_values_plus[i] = value
        else:
            left_hand_values_minus[i] = value

df['+LHS'] = left_hand_values_plus
df['-LHS'] = left_hand_values_minus

#df.to_csv("Neutron Spins & Scattering Lengths - Sheet1 (2).csv", sep=',', index=False)

df2 = pd.read_csv('H through Ca Neutron Spins & Scattering Lengths - Sheet1 (2) - Neutron Spins & Scattering Lengths - Sheet1 (2).csv', sep=',')

Vplus = np.empty(len(df2['Isotope']))
Vminus = np.empty(len(df2['Isotope']))
Vnaught = np.empty(len(df2['Isotope']))
Vspin = np.empty(len(df2[['Isotope']]))

for i in range(len(df2['Isotope'])):
    Vplus[i] = well_depth([df2['Solution+'][i], df2['A'][i]])
    Vminus[i] = well_depth([df2['Solution-'][i], df2['A'][i]])
    if df2['Spin'][i] == 0:
        Vspin[i] = 0
        Vnaught[i] = Vplus[i]
    else:
        coeffs_array = np.array([[1, df2['Spin'][i]/2], [1, -0.5*(df2['Spin'][i]+1)]])
        sols = np.array([Vplus[i], Vminus[i]])
        new_depths = np.linalg.solve(coeffs_array, sols)
        Vnaught[i] = new_depths[0]
        Vspin[i] = new_depths[1]

df2['V+'] = Vplus
df2['V-'] = Vminus
df2['V0'] = Vnaught
df2['Vs'] = Vspin