import numpy as np

def uncertainty_prop(std_value, params, errors, func):
    var_error = np.zeros(len(params))
    for i in range(len(params)):
        for j in range(2):
            if j == 1:
                errors[i] *= -1
            new_param = params[i] + errors[i]
            original_param = params[i]
            params[i] = new_param
            var_error[i] += np.abs(func(params) - std_value)
            params[i] = original_param
    var_error /= 2
    square_sum = 0
    for i in range(len(var_error)):
        square_sum += var_error[i]**2
    uncertainty = np.sqrt(square_sum)
    return uncertainty

def uncertainties_off(expected, measured, error):
    return np.abs((expected - measured) / error)
