### LSST Emulator by me :)
import os
import sys
import numpy as np
import torch
from torch import nn

#sys.path.append(os.path.join(os.path.dirname("__file__"), '../../LSST_emulation'))
#from cocoa_emu import *
#from cocoa_emu.emulator import NNEmulator, GPEmulator
#from cocoa_emu.data_model import LSST_3x2
sys.path.append(os.path.join(os.path.dirname("__file__"), '../../'))
import cocoa_emu
from cocoa_emu import Config, nn_pca_emulator
#import cosmopower.likelihoods.tf_planck2018_lite as cppl
#import cosmopower as cp

class lsst_emulator:
    def __init__(self,
                 cosmo_prior_lim=None,
                 ia_prior_lim=None,
                 bias_prior_lim=None,
                 baryon_prior_lim=None,
                 dz_source_std=None,
                 dz_lens_std=None,
                 shear_calib_std=None,
                 dv_fid=None,
                 cov=None,
                 bias_fid=None,
                 bias_mask=None,
                 shear_calib_mask=None,
                 data_model=None,
                 emu=None
                ):
        
        self.cosmo_prior_lim = np.array([[1.61, 3.91],
                               [0.87, 1.07],
                               [55, 91],
                               [0.01, 0.04],
                               [0.001, 0.99]])

        self.ia_prior_lim = np.array([[-5., 5.],
                               [-5., 5.]])

        self.dz_source_std   = 0.002 * np.ones(5)
        self.shear_calib_std = 0.005 * np.ones(5)

        configfile = '/home/grads/data/evan/train_emulator.yaml'
        config = Config(configfile)

        OUTPUT_DIM = 780
        self.mask             = config.mask[:OUTPUT_DIM]
        cov                   = config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM]
        cov_inv               = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM])
        self.cov_inv_masked   = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM][self.mask][:,self.mask])
        self.shear_calib_mask = config.shear_calib_mask[:,:OUTPUT_DIM]

        # set needed parameters to initialize emulator
        device=torch.device('cpu')
        evecs=0
        self.emu = nn_pca_emulator(nn.Sequential(nn.Linear(1,1)), config.dv_fid, config.dv_std, cov_inv, evecs, device)
        self.emu.load('/home/grads/data/evan/attention_transformer_test_500_epochs')
        self.emu.model.double()
        self.dv_fid = config.dv_fid
        
#     def add_bias(self, bias_theta, datavector):
#         for i in range(5):
#             factor = (bias_theta[i] / self.bias_fid[i])**self.bias_mask[i]
#             datavector = factor * datavector
#         return datavector

    def add_shear_calib(self, m, datavector):
        for i in range(5):
            factor = (1 + m[i])**self.shear_calib_mask[i]
            datavector = factor * datavector
        return datavector

    def hard_prior(self, theta, params_prior):
        """
        A function to impose a flat prior on a set of parameters.
        :theta: The set of parameter values
        :params_prior: The minimum and the maximum value of the parameters on which this prior is imposed
        """
        is_lower_than_min = bool(np.sum(theta < params_prior[:,0]))
        is_higher_than_max = bool(np.sum(theta > params_prior[:,1]))
        if is_lower_than_min or is_higher_than_max:
            return -np.inf
        else:
            return 0.
        
    def log_prior(self, theta):
        cosmo_theta = theta[:5]
        ns          = cosmo_theta[1]
        ns_prior    = 0.

        dz_source   = theta[5:10]
        ia_theta    = theta[10:12]
        shear_calib = theta[12:17]
        #dz_lens     = theta[12:17]
        #bias        = theta[17:22]
        #shear_calib = theta[22:27]
        #baryon_q    = theta[27:]

        cosmo_prior = self.hard_prior(cosmo_theta, self.cosmo_prior_lim) + ns_prior
        ia_prior    = self.hard_prior(ia_theta, self.ia_prior_lim)
        #bias_prior  = self.hard_prior(bias, self.bias_prior_lim)
        #baryon_prior = self.hard_prior(baryon_q, self.baryon_prior_lim)

        dz_source_lnprior   = -0.5 * np.sum((dz_source / self.dz_source_std)**2)
        #dz_lens_lnprior     = -0.5 * np.sum((dz_lens / self.dz_lens_std)**2)
        shear_calib_lnprior = -0.5 * np.sum((shear_calib / self.shear_calib_std)**2)

        return cosmo_prior + ia_prior + dz_source_lnprior + shear_calib_lnprior#+ dz_lens_lnprior + \ #+ bias_prior + baryon_prior

#     def get_data_vector_emu(self, theta):
#         """
#         Function to get the emulated data vector (including the effect of galaxy bias, baryons, etc.)
#         """
#         cosmo_ia_dz_theta = theta[:12]
#         shear_calib = theta[12:17]
#         #bias        = theta[17:22]
#         #shear_calib = theta[22:27]
#         #baryon_q    = theta[27:]
#         datavector = self.data_model.compute_datavector(cosmo_ia_dz_theta)
#         datavector = np.array(datavector)
#         datavector = self.add_bias(bias, datavector)
#         datavector = self.add_shear_calib(shear_calib, datavector)
#         return datavector
    
    def ln_lkl(self,theta):
        param = torch.Tensor(theta[:12])
        shear = theta[12:17]
        dv_cs = self.emu.predict(param)[0]
        dv = self.add_shear_calib(shear,dv_cs)
        dv_diff_masked = (dv - self.dv_fid[:780])[self.mask]
        lkl = -0.5 * dv_diff_masked @ self.cov_inv_masked @ dv_diff_masked
        return lkl

    def log_prob(self, theta):
        return self.log_prior(theta) + self.ln_lkl(theta)