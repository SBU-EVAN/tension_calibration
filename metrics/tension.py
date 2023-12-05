import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tf.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import scipy

def flow_significance(trained_dist,bijector,nparams,points_per_try=1000,alpha=0.32,tol=1e-2):
    # The alpha is used for beta function for the confidence. Raveri et. al. defaults to 0.32
    prob = trained_dist.prob(bijector.inverse(np.zeros(nparams,dtype=np.float32)))
    n_pass = 0

    n_points = 0

    sigma_low  = 0
    sigma_high = 100

    while (sigma_high-sigma_low)>=2*tol:
        n_points+=points_per_try

        _s = trained_dist.sample(points_per_try)
        _v = trained_dist.prob(_s)
        for val in _v:
            if val>prob:
                n_pass+=1
        # use clopper-pearson to find confidence level
        low = scipy.stats.beta.ppf(alpha/2,float(n_pass),float(n_points-n_pass+1))
        high = scipy.stats.beta.ppf(1-alpha/2,float(n_pass+1),float(n_points-n_pass))
        sigma_high = np.sqrt(2)*scipy.special.erfinv(high)
        sigma_low = np.sqrt(2)*scipy.special.erfinv(low)
        if n_points>=1e7:
            print('n_points exceed 10^7! giving up...')
            break

    # compute sigma based on gaussian
    n_sigma = np.sqrt(2)*scipy.special.erfinv(n_pass/n_points)
    sigma_high = np.sqrt(2)*scipy.special.erfinv(high)
    sigma_low = np.sqrt(2)*scipy.special.erfinv(low)

    return n_sigma,sigma_high,sigma_low
