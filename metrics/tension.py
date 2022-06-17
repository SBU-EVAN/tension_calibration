import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tf.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import scipy

def flow_significance(trained_dist,bijector,nparams,n_points = 100000,alpha=0.32):
    # The alpha is used for beta function for the confidence. Raveri et. al. defaults to 0.32
    prob = trained_dist.prob(bijector.inverse(np.zeros(nparams,dtype=np.float32)))
    n_pass = 0

    _s = trained_dist.sample(n_points)
    _v = trained_dist.prob(_s)
    for val in _v:
        if val>prob:
            n_pass+=1
    # use clopper-pearson to find confidence level
    low = scipy.stats.beta.ppf(alpha/2,float(n_pass),float(n_points-n_pass+1))
    high = scipy.stats.beta.ppf(1-alpha/2,float(n_pass+1),float(n_points-n_pass))

    # compute sigma based on gaussian
    n_sigma = np.sqrt(2)*scipy.special.erfinv(n_pass/n_points)
    sigma_high = np.sqrt(2)*scipy.special.erfinv(high)
    sigma_low = np.sqrt(2)*scipy.special.erfinv(low)

    return n_sigma,sigma_high,sigma_low
