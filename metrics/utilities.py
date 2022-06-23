import scipy
import numpy as np

def optimizer(func,dim,init_guess=None):
    # func: callable - function to be minimized.
    # dim = number of dimensions
    # prior_ranges = boundary of the domain to optimize.
    #
    # return: optimizer result.
    if init_guess is None:
        init_guess = np.zeros(dim)

    # minimize -prob
    MAP = scipy.optimize.minimize(func,
                                  init_guess
                                 )
    theta = MAP.x
    return theta