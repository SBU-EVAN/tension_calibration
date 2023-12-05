import sys
import os
import numpy as np
import getdist
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname("__file__"), '../'))
from metrics import diff
from metrics import flow
from metrics import tension
from metrics.parameter_metrics import *
from metrics import utilities

def compute_omega_b_c(pars):
    params   = pars[:,:5]
    H0       = params[:,2]
    omegab   = params[:,3]
    omegam   = params[:,4]
    omeganh2 = (3.046/3)**(3/4)*0.06/94.1

    h = H0/100

    omegabh2 = omegab*(h**2)
    omegach2 = (omegam-omegab)*(h**2) - omeganh2

    return(omegabh2,omegach2)

# Paths for chains
lsst_path    = '/home/grads/extra_data/evan/shifted_chains/'
planck_path  = '/home/grads/extra_data/evan/planck_chains/'
joint_path = '/home/grads/extra_data/evan/shifted_joint_chains/'

#priors
priors = {'omegab':  [0.001, 0.04],
          'omegac':  [0.005, 0.99],
          'omegabh2':  [0.001, 0.04],
          'omegach2':  [0.005, 0.99],
          'h':       [0.2,   1.0],
          'H0':      [55,    91],
          'tau':     [0.01,  0.8],
          'ns':      [0.9,   1.1],
          'logAs':    [1.61,  3.91],
          'logA':    [1.61,  3.91],
          'Aplanck': [1.0,   0.01],
         }

start = 0 
stop = 1
shift = 5   # the shift to consider
results = []

if shift==5:
    lsst_name   = 'shift_5_fid'
    joint_name  = 'joint_lkl_p5'
    lkl_b    = -1.0192232691002423
    lkl_ab   = -291.31686
if shift==0:
    lsst_name   = 'lsst_at_planck_test_mobo'
    joint_name  = 'joint_lkl_0'
    lkl_b    = -0.6186452962402702
    lkl_ab   = -292.25833
if shift==-5:
    lsst_name   = 'shift_-5_fid'
    joint_name  = 'joint_lkl_m5'
    lkl_b    = -0.8548166697019856
    lkl_ab   = -294.21417

#open optimizer results
# planck_lkl = np.loadtxt('/home/grads/data/evan/emulator/planck_optimizer_lkl.txt')
# lsst_lkl   = np.loadtxt('/home/grads/data/evan/emulator/lsst_optimizer_lkl_'+shift_str+'.txt')
# joint_lkl  = np.loadtxt('/home/grads/data/evan/emulator/joint_optimizer_lkl_'+shift_str+'.txt')

### Now run the metrics
try:
    planck_chain = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/cosmopower_emcee/notebooks/planck_emulated',no_cache=True)
    lsst_chain   = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/emulator/'+lsst_name,no_cache=True)
    joint_chain  = getdist.mcsamples.loadMCSamples('/home/grads/extra_data/evan/fiducial_chains/'+joint_name,no_cache=True)
except:
    print('A chain is missing!')

samples_1 = lsst_chain.samples
omega_b,omega_c = compute_omega_b_c(samples_1)

samples_1[:,3] = omega_b
samples_1[:,4] = omega_c

lsst_chain.setSamples(samples_1)

samples_2 = joint_chain.samples
omega_b,omega_c = compute_omega_b_c(samples_2)

samples_2[:,3] = omega_b
samples_2[:,4] = omega_c

joint_chain.setSamples(samples_2)

#need to convert h to H0
s = planck_chain.samples
s[...,2] = 100*s[...,2] 
planck_chain.setSamples(s)

#rename planck chain
planck_chain.setParamNames(['omegab','omegac','H0','tau','ns','logAs','Aplanck'])
lsst_chain.setParamNames(['logAs','ns','H0','omegab','omegac','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5'])
joint_chain.setParamNames(['logAs','ns','H0','omegab','omegac','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5','tau','Aplanck'])

planck_samples = planck_chain.samples[:,[5,4,2,0,1]]
lsst_samples   = lsst_chain.samples[:,:5]
joint_samples  = joint_chain.samples[:,:5]

chain3 = getdist.mcsamples.MCSamples(samples=planck_samples,names=['logAs','ns','H0','omegab','omegac'])
chain1 = getdist.mcsamples.MCSamples(samples=lsst_samples,names=['logAs','ns','H0','omegab','omegac'])
chain2 = getdist.mcsamples.MCSamples(samples=joint_samples,names=['logAs','ns','H0','omegab','omegac'])

lkl_a  = -2.887738037109375000e+02 #PLANCK

nsigma = q_dmap(chain3,chain1,chain2,prior_dict=priors,lkl_a=lkl_a,lkl_b=lkl_b,lkl_ab=lkl_ab)
print(nsigma)

print('done!')



###########################################################################
#
# OPTIMIZER RUNS:
#
### LSST +5 ###############################################################
#
# PARAMS = [ 3.02006753e+00  9.52811353e-01  6.98608789e+01  4.58908890e-02
#            3.06520832e-01  2.85251441e-03  1.10005216e-04 -2.10671647e-04
#           -3.16028043e-04  1.38059494e-04  4.33577728e-01 -9.92347810e-01
#            3.64735543e-04  7.82385020e-04  7.15436149e-04  1.94140754e-04
#           -1.63514536e-03]
#
# LKL = -1.0192232691002423
#
###########################################################################

### LSST 0  ###############################################################
#
# PARAMS = [ 3.01500013e+00  9.54737688e-01  7.05292169e+01  4.95445568e-02
#            3.04913291e-01  3.02871697e-03  9.71285124e-05 -2.38961203e-04
#           -3.61646372e-04  1.14447015e-04  4.54201216e-01 -8.86503131e-01
#            3.10774991e-04  4.35364191e-04  3.31006515e-04  7.30392148e-05
#           -9.97567527e-04]
#
# LKL = -0.6186452962402702
#
###########################################################################

### LSST -5 ###############################################################
#
# PARAMS = [ 3.03470010e+00  9.50002123e-01  7.23495622e+01  5.01262258e-02
#            2.92424023e-01  2.44053389e-03  9.20366268e-05 -1.43454164e-04
#           -2.13879103e-04  9.60933945e-05  4.29753899e-01 -9.97803150e-01
#            3.33660760e-04  6.49590021e-04  7.18163641e-04  2.61633366e-04
#           -1.57748802e-03]
# 
# LKL = -0.8548166697019856
#
###########################################################################



### JOINT -5 ##############################################################
#
# PARAMS = [ 3.07488919e+00  9.66015832e-01  6.78320002e+01  4.87400157e-02
#            3.11677181e-01  2.10315669e-03  2.11022162e-04 -4.43715067e-06
#           -1.68433230e-04 -8.14328798e-05  4.83318090e-01 -4.55705593e-01
#            3.11228900e-04  3.79243400e-04  1.12353646e-04 -1.75195040e-04
#           -1.06684009e-03  6.76849639e-02  1.00157417e+00]
#
# LKL = [-291.31686]
#
###########################################################################

### JOINT 0  ##############################################################
#
# PARAMS = [ 3.05669190e+00  9.67146541e-01  6.82253226e+01  4.83038741e-02
#            3.06322614e-01  9.81388790e-04 -1.17502429e-05 -1.06717849e-04
#           -2.73366074e-05  3.46085957e-05  4.77874060e-01 -3.42222056e-01
#            3.20543378e-04  3.01395323e-04  1.81956803e-04 -8.03815640e-05
#           -6.87082305e-04  6.24794614e-02  9.98656151e-01]
#
# LKL = [-292.25833]
#
###########################################################################

### JOINT -5 ##############################################################
#
# PARAMS = [ 3.03908430e+00  9.68883288e-01  6.85506159e+01  4.79245016e-02
#            3.01839796e-01 -4.48065667e-04 -1.91875423e-04 -9.36905780e-05
#           -3.27475479e-04  7.03488739e-07  4.94741614e-01  2.07460565e-02
#           -2.37383158e-04  8.77357644e-05 -5.39013999e-04 -3.36549605e-04
#           -6.50372154e-04  5.88440472e-02  9.94549443e-01]
#
# LKL = [-294.21417]
#
###########################################################################

# PLANCK LKL = -2.887738037109375000e+02
