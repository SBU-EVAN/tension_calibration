### LSST Emulator by Supranta et. al.
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname("__file__"), '../../LSST_emulation'))
from cocoa_emu import *
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.data_model import LSST_3x2

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
                 fid=None,
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

        self.bias_prior_lim = np.array([[0.8, 3.],
                               [0.8, 3.],
                               [0.8, 3.],
                               [0.8, 3.],
                               [0.8, 3.]])

        self.baryon_prior_lim = np.array([[-3., 12.],
                                     [-2.5, 2.5]])

        self.baryon_prior_lim = 3. * self.baryon_prior_lim 

        self.dz_source_std   = 0.002 * np.ones(5)
        self.dz_lens_std     = 0.005 * np.ones(5)
        self.shear_calib_std = 0.005 * np.ones(5)
        
        # Get the LSST covariance and fid data
        path = '/home/grads/data/evan/LSST_emulation/data/lsst_y1/'
        lsst_cov = np.loadtxt(path+'cov_lsst_y1')
        fid_cos = np.loadtxt(path+'lsst_y1_data_fid',dtype=np.float32)[:,1]

        lsst_y1_cov = np.zeros((1560, 1560))
        for line in lsst_cov:
            i = int(line[0])
            j = int(line[1])

            cov_g_block  = line[-2]
            cov_ng_block = line[-1]

            cov_ij = cov_g_block + cov_ng_block

            lsst_y1_cov[i,j] = cov_ij
            lsst_y1_cov[j,i] = cov_ij

        self.fid = torch.Tensor(fid_cos)
        self.cov = torch.Tensor(lsst_y1_cov)

        # Code taken from the emulator notebook
        #first the fiducial cosmology

        configfile = '/home/grads/data/evan/LSST_emulation/configs/nn_emu.yaml'
        config = Config(configfile)

        config_args     = config.config_args
        config_args_io  = config_args['io']
        config_args_data = config_args['data']

        savedir = '/home/grads/data/evan/LSST_emulation/output/nn_emu/'

        N_DIM         = 17
        self.data_model    = LSST_3x2(N_DIM, config_args_io, config_args_data)
        self.data_model.emu_type = 'nn'
        OUTPUT_DIM = 1560

        self.emu = NNEmulator(N_DIM, OUTPUT_DIM, self.data_model.dv_fid, self.data_model.dv_std)    
        self.emu.load('/home/grads/data/evan/LSST_emulation/model/nn_emu/model')
        # ======================================================

        self.data_model.emu = self.emu

        self.bias_fid         = self.data_model.bias_fid
        self.bias_mask        = self.data_model.bias_mask
        self.shear_calib_mask = self.data_model.shear_calib_mask
        
    
    def add_bias(self, bias_theta, datavector):
        for i in range(5):
            factor = (bias_theta[i] / self.bias_fid[i])**self.bias_mask[i]
            datavector = factor * datavector
        return datavector

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
        dz_lens     = theta[12:17]
        bias        = theta[17:22]
        shear_calib = theta[22:27]
        baryon_q    = theta[27:]

        cosmo_prior = self.hard_prior(cosmo_theta, self.cosmo_prior_lim) + ns_prior
        ia_prior    = self.hard_prior(ia_theta, self.ia_prior_lim)
        bias_prior  = self.hard_prior(bias, self.bias_prior_lim)
        baryon_prior = self.hard_prior(baryon_q, self.baryon_prior_lim)

        dz_source_lnprior   = -0.5 * np.sum((dz_source / self.dz_source_std)**2)
        dz_lens_lnprior     = -0.5 * np.sum((dz_lens / self.dz_lens_std)**2)
        shear_calib_lnprior = -0.5 * np.sum((shear_calib / self.shear_calib_std)**2)

        return cosmo_prior + ia_prior + dz_source_lnprior + dz_lens_lnprior + \
                shear_calib_lnprior + bias_prior + baryon_prior

    def get_data_vector_emu(self, theta):
        """
        Function to get the emulated data vector (including the effect of galaxy bias, baryons, etc.)
        """
        cosmo_ia_dz_theta = theta[:17]
        bias        = theta[17:22]
        shear_calib = theta[22:27]
        baryon_q    = theta[27:]
        datavector = self.data_model.compute_datavector(cosmo_ia_dz_theta)
        datavector = np.array(datavector)
        datavector = self.add_bias(bias, datavector)
        datavector = self.add_shear_calib(shear_calib, datavector)
        return datavector
    
    def log_lkl(self, theta):
        model_datavector = self.get_data_vector_emu(theta)[0]
        delta_dv = (model_datavector - self.data_model.dv_obs)[self.data_model.mask_3x2]
        return -0.5 * delta_dv @ self.data_model.masked_inv_cov @ delta_dv

    def log_prob(self, theta):
        return self.log_prior(theta) + self.log_lkl(theta)