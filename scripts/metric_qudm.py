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
joint_path   = '/home/grads/extra_data/evan/shifted_joint_chains/'
planck_path  = '/home/grads/extra_data/evan/planck_chains/'
savename     = 'planck_lsst_nf_m5.txt'

planck_chain = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/cosmopower_emcee/notebooks/planck_emulated',no_cache=True)
# s = planck_chain.samples
# s[...,2] = 100*s[...,2] 
# planck_chain.setSamples(s)
h = planck_chain.getParams().h
omegabh2 = planck_chain.getParams().omegab
omegach2 = planck_chain.getParams().omegac

s = planck_chain.samples
s[:,0] = omegabh2/h**2
planck_chain.setSamples(s)
#planck_chain.addDerived(h*100,'H0')
omeganh2 = (3.046/3)**(3/4)*0.06/94.1
omegamh2 = omegach2+omeganh2
omegam   = omegamh2/(h**2)+omegabh2
s[:,1]=omegam
s[:,2] = h*100
print(s[0])
planck_chain.setParamNames(['omegab','omegac','H0','tau','ns','logAs','aplanck'])

start = 0   # range to run results
stop  = 256 # the number of noise realizations
shift = 0   # the shift to consider
results = []

if shift==5:
    savename  = 'qudm_p5.txt' #'planck_lsst_nf_p5.txt'
    shift_str = 'p5'
if shift==0:
    savename  = 'qudm_0.txt' #'planck_lsst_nf_0.txt'
    shift_str = '0'
if shift==-5:
    savename  = 'qudm_m5.txt' #'planck_lsst_nf_m5.txt'
    shift_str = 'm5'

for i in range(start,stop):
    try:
        joint_chain = getdist.mcsamples.loadMCSamples(joint_path+'joint_lkl_'+str(shift_str)+'_'+str(i),no_cache=True)
    except:
        print('{} not found'.format(i))
        results.append(-1)
        continue

    # get omegabh2, omegach2 for joint
    samples = joint_chain.samples
    # omega_b,omega_c = compute_omega_b_c(samples)

    # samples[:,3] = omega_b
    # samples[:,4] = omega_c

    joint_chain.setSamples(samples)
    joint_chain.setParamNames(['logAs','ns','H0','omegab','omegac','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5','tau','Aplanck'])

    # Get rid of extra parameters
    joint_samples = joint_chain.samples[:,:5]

    #chain1 = getdist.mcsamples.MCSamples(samples=lsst_samples)
    chain2 = getdist.mcsamples.MCSamples(samples=joint_samples,names=['logAs','ns','H0','omegab','omegac'])

    ### Parameter Update
    nsigma = qudm(planck_chain,chain2,feedback=False)
    print(i,nsigma)
    results.append(nsigma)

    if i%10==0:
        np.savetxt(savename,results)
        print('checkpointed')

np.savetxt(savename,results)
print('done!')
