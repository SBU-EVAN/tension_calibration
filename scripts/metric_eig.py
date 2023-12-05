import sys
import os
import numpy as np
import getdist
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

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

def compute_omega_b_m(pars):
    params   = pars[:,:6]
    h        = pars[:,2]/100
    omegabh2 = pars[:,0]
    omegach2 = pars[:,1]

    omeganh2 = (3.046/3)**(3/4)*0.06/94.1

    omegamh2 = omegabh2+omegach2+omeganh2

    omegab = omegabh2/h**2
    omegam = omegamh2/h**2

    return(omegab,omegam)

#priors
priors = {'omegab':  [0.03, 0.07],
          'omegac':  [0.005, 0.99],
          'omegam':  [0.1, 0.9],
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

# Paths for chains
lsst_path    = '/home/grads/extra_data/evan/shifted_chains/'
planck_path  = '/home/grads/extra_data/evan/planck_chains/'

start = 0   # range to run results
stop  = 256 # the number of noise realizations
shift = 0   # the shift to consider
results = []

if shift==5:
    savename = 'eig_p5.txt' #'planck_lsst_nf_p5.txt'
if shift==0:
    savename = 'eig_0.txt' #'planck_lsst_nf_0.txt'
if shift==-5:
    savename = 'eig_m5.txt' #'planck_lsst_nf_m5.txt'

for i in range(start,stop):
    planck_chain = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/cosmopower_emcee/notebooks/planck_emulated',no_cache=True)#planck_path+str(i))
    try:
        lsst_chain   = getdist.mcsamples.loadMCSamples(lsst_path+'shift_'+str(shift)+'_noise'+str(i),no_cache=True)
    except:
        print('{} not found')
        results.append(-1)
        continue
        
    # samples = lsst_chain.samples
    # omega_b,omega_c = compute_omega_b_c(samples)

    # samples[:,3] = omega_b
    # samples[:,4] = omega_c

    # lsst_chain.setSamples(samples)
    
    #need to convert h to H0
    s = planck_chain.samples
    s[...,2] = 100*s[...,2] 
    omb,omm = compute_omega_b_m(s)
    s[...,0]=omb
    s[...,1]=omm
    planck_chain.setSamples(s)
    print(planck_chain.getMeans())
    lsst_chain.thin(3)

    #rename planck chain
    planck_chain.setParamNames(['omegab','omegam','H0','tau','ns','logAs','Aplanck'])
    lsst_chain.setParamNames(['logAs','ns','H0','omegab','omegam','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5'])
    
    print()
    
    ### Normalizing flow
    nsigma,high,low = eigentension(lsst_chain,planck_chain,priors,feedback=True)
    results.append(nsigma)
    if( i%10==0 ):
        np.savetxt(savename,np.array(results))
        print('Checkpointed!')

np.savetxt(savename,np.array(results))
print('done!')
