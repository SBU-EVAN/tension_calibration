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
priors = {'logAs':  [1.61, 3.91],
          'ns':     [0.87, 1.07],
          'H0':     [55,   91]  ,
          'omegab': [0.03, 0.07],
          'omegam': [0.1,  0.9] ,
         }

# Paths for chains
planck_path  = '/home/grads/extra_data/evan/planck_chains/'

lsst_path = '/home/grads/data/evan/emulator/'
chain_list = ['shift_5_fid','lsst_at_planck_test_mobo','shift_-5_fid']

for i,chain in enumerate(chain_list):
    planck_chain = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/cosmopower_emcee/notebooks/planck_emulated',no_cache=True)#planck_path+str(i))
    lsst_chain   = getdist.mcsamples.loadMCSamples(lsst_path+chain,no_cache=True)
    #planck_chain   = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/emulator/cmb_guassian')

    #planck_chain.setParamNames(['logAs','ns','H0','omegab','omegam'])#['omegab','omegac','H0','tau','ns','logAs','Aplanck'])
    planck_chain.setParamNames(['omegab','omegam','H0','tau','ns','logAs','Aplanck'])

    # samples = lsst_chain.samples
    # omega_b,omega_c = compute_omega_b_c(samples)

    # samples[:,3] = omega_b
    # samples[:,4] = omega_c

    # lsst_chain.setSamples(samples)
    
    #need to convert h to H0
    s = planck_chain.samples
    print(s[0])
    s[...,2] = 100*s[...,2] 
    omb,omm = compute_omega_b_m(s)
    s[...,0]=omb
    s[...,1]=omm
    print(s[0])
    planck_chain.setSamples(s)

    #rename planck chain
    #planck_chain.setParamNames(['omegab','omegac','H0','tau','ns','logAs','Aplanck'])
    lsst_chain.setParamNames(['logAs','ns','H0','omegab','omegam','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5'])
    
    print()
    
    ### Normalizing flow
    nsigma,high,low = eigentension(lsst_chain,planck_chain,priors,feedback=True)
    print(nsigma,chain)

print('done!')
