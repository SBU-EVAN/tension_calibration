import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import os
from numpy import linalg
import cosmopower as cp
sys.path.append('/home/grads/data/evan/cosmopower_emcee/cosmopower')
import likelihoods.tf_planck2018_lite as cppl

class cosmopower:
    def __init__(self,
                 parameters_and_priors=None,
                 tf_planck=None
                ):
        # open the models
        tt_emu_model = cp.cosmopower_NN(restore=True,
                                        restore_filename=\
                                        '/home/grads/data/evan/cosmopower_emcee/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN'
                                       )

        te_emu_model = cp.cosmopower_PCAplusNN(restore=True,
                                        restore_filename=\
                                        '/home/grads/data/evan/cosmopower_emcee/cosmopower/trained_models/CP_paper/CMB/cmb_TE_PCAplusNN'
                                       )

        ee_emu_model = cp.cosmopower_NN(restore=True,
                                        restore_filename=\
                                        '/home/grads/data/evan/cosmopower_emcee/cosmopower/trained_models/CP_paper/CMB/cmb_EE_NN'
                                       )
        
        # path to the tf_planck2018_lite likelihood
        tf_planck2018_lite_path = '/home/grads/data/evan/cosmopower_emcee/cosmopower/likelihoods/tf_planck2018_lite/'

        # parameters of the analysis, and their priors
        self.parameters_and_priors = {'omega_b':      [0.001, 0.04, 'uniform'],
                                      'omega_cdm':    [0.005, 0.99,  'uniform'],
                                      'h':            [0.2,   1.0,   'uniform'],
                                      'tau_reio':     [0.01,  0.8,   'uniform'],
                                      'n_s':          [0.9,   1.1,   'uniform'],
                                      'ln10^{10}A_s': [1.61,  3.91,  'uniform'],
                                      'A_planck':     [1.0,   0.01,  'gaussian'],
                                      }

        # instantiation
        self.tf_planck = cppl(parameters=self.parameters_and_priors, 
                              tf_planck2018_lite_path=\
                              '/home/grads/data/evan/cosmopower_emcee/cosmopower/likelihoods/tf_planck2018_lite',
                              tt_emu_model=tt_emu_model,
                              te_emu_model=te_emu_model,
                              ee_emu_model=ee_emu_model
                             )
        
    def log_prob(self, theta):
        p=self.tf_planck.posterior(theta.astype(np.float32)).numpy()
        return p
    
    def log_lkl(self, theta):
        lkl = self.tf_planck.get_loglkl(theta.astype(np.float32)).numpy()
        return lkl