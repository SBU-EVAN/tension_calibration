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
savename = 'new_emu_constant_planck_results.txt'

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
stop = 256
shift = 0   # the shift to consider
results = []

if shift==5:
    savename  = 'qdmap_p5.txt' #'planck_lsst_nf_p5.txt'
    shift_str = 'p5'
if shift==0:
    savename  = 'qdmap_0.txt' #'planck_lsst_nf_0.txt'
    shift_str = '0'
if shift==-5:
    savename  = 'qdmap_m5.txt' #'planck_lsst_nf_m5.txt'
    shift_str = 'm5'

#open optimizer results
planck_lkl = np.loadtxt('/home/grads/data/evan/emulator/planck_optimizer_lkl.txt')
lsst_lkl   = np.loadtxt('/home/grads/data/evan/emulator/lsst_optimizer_lkl_'+shift_str+'.txt')
joint_lkl  = np.loadtxt('/home/grads/data/evan/emulator/joint_optimizer_lkl_'+shift_str+'.txt')

### Now run the metrics
for i in range(start,stop):
    try:
        planck_chain = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/cosmopower_emcee/notebooks/planck_emulated',no_cache=True)
        print('planck')
        lsst_chain   = getdist.mcsamples.loadMCSamples(lsst_path+'shift_'+str(shift)+'_noise'+str(i),no_cache=True)
        print('lsst')
        joint_chain  = getdist.mcsamples.loadMCSamples(joint_path+'joint_lkl_'+str(shift_str)+'_'+str(i),no_cache=True)
    except:
        print('A chain is missing!')
        continue

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

    lkl_a  = planck_lkl
    lkl_b  = lsst_lkl[i]
    lkl_ab = joint_lkl[i]

    print(lkl_a)
    print(lkl_b)
    print(lkl_ab)
    
    nsigma = q_dmap(chain3,chain1,chain2,prior_dict=priors,lkl_a=lkl_a,lkl_b=lkl_b,lkl_ab=lkl_ab)
    results.append(nsigma)
    
    np.savetxt(savename,results)

print('done!')
