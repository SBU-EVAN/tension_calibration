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

shift = 0   # the shift to consider
shifts = [-5,0,5]
planck_chain = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/cosmopower_emcee/notebooks/planck_emulated',no_cache=True)

h = planck_chain.getParams().h
omegabh2 = planck_chain.getParams().omegab
omegach2 = planck_chain.getParams().omegac
s = planck_chain.samples
print(s[0])
s[:,0] = omegabh2/h**2
planck_chain.setSamples(s)

omeganh2 = (3.046/3)**(3/4)*0.06/94.1
omegamh2 = omegach2+omeganh2
omegam   = omegamh2/(h**2)+omegabh2
s[:,1] = omegam
s[:,2] = 100*h

planck_chain.setSamples(s)
print(s[0,:6])

for shift in shifts:
    if shift==5:
        lsst_name   = 'shift_5_fid'
        joint_name  = 'joint_lkl_p5'
    if shift==0:
        lsst_name   = 'lsst_at_planck_test_mobo'
        joint_name  = 'joint_lkl_0'
    if shift==-5:
        lsst_name   = 'shift_-5_fid'
        joint_name  = 'joint_lkl_m5'

    try:
        #lsst_chain   = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/emulator/'+lsst_name,no_cache=True)
        joint_chain  = getdist.mcsamples.loadMCSamples('/home/grads/extra_data/evan/fiducial_chains/'+joint_name,no_cache=True)
    except:
        print('{} not found'.format(i))

    # get omegabh2, omegach2 for lsst
    #samples = lsst_chain.samples
    #omega_b,omega_c = compute_omega_b_c(samples)

    #samples[:,3] = omega_b
    #samples[:,4] = omega_c

    #lsst_chain.setSamples(samples)

    # get omegabh2, omegach2 for joint
    # samples = joint_chain.samples
    # omega_b,omega_c = compute_omega_b_c(samples)

    # samples[:,3] = omega_b
    # samples[:,4] = omega_c

    # joint_chain.setSamples(samples)
    print(joint_chain.samples[0,:5])

    #lsst_chain.setParamNames(['logAs','ns','H0','omegab','omegac','ldz1','ldz2','ldz3','ldz4','ldz5','lIA1','lIA2','lm1','lm2','lm3','lm4','lm5'])
    joint_chain.setParamNames(['logAs','ns','H0','omegab','omegac','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5','tau','Aplanck'])

    # Get rid of extra parameters
    #lsst_samples  = lsst_chain.samples[:,:5]
    joint_samples = joint_chain.samples[:,:5]
    # print(joint_samples[0])
    # print(planck_chain.samples[0,[5,4,2,0,1]])

    #chain1 = getdist.mcsamples.MCSamples(samples=lsst_samples)
    chain2 = getdist.mcsamples.MCSamples(samples=joint_samples,names=['logAs','ns','H0','omegab','omegac'])

    ### Parameter Update
    nsigma = qudm(planck_chain,chain2,feedback=True)
    print(shift,nsigma)
    print('done!')
