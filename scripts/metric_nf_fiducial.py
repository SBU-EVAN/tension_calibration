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

def compute_omega_b_c_planck(pars):
    params   = pars[:,:6]
    H0       = params[:,2]
    omegab   = params[:,0]
    omegam   = params[:,1]
    omeganh2 = (3.046/3)**(3/4)*0.06/94.1

    h = H0/100

    omegabh2 = omegab*(h**2)
    omegach2 = (omegam-omegab)*(h**2) - omeganh2

    return(omegabh2,omegach2)


# Paths for chains
#lsst_path    = '/home/grads/extra_data/evan/shifted_chains/'
planck_path  = '/home/grads/extra_data/evan/planck_chains/'

lsst_path = '/home/grads/data/evan/emulator/'
chain_list = ['shift_5_fid','lsst_at_planck_test_mobo','shift_-5_fid']

for i,chain in enumerate(chain_list):
    #planck_chain = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/cosmopower_emcee/notebooks/planck_emulated',no_cache=True)#planck_path+str(i))
    #planck_chain = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/cosmopower_emcee/notebooks/plank_omb_omm',no_cache=True)
    planck_chain   = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/emulator/cmb_guassian')
    lsst_chain   = getdist.mcsamples.loadMCSamples(lsst_path+chain,no_cache=True)
        
    #samples = lsst_chain.samples
    #omega_b,omega_c = compute_omega_b_c(samples)

    #samples[:,3] = omega_b
    #samples[:,4] = omega_c

    #lsst_chain.setSamples(samples)
    
    #need to convert h to H0
    #s = planck_chain.samples
    #omega_b,omega_c = compute_omega_b_c_planck(s)

    #s[:,0] = omega_b
    #s[:,1] = omega_c
    #planck_chain.setSamples(s)

    #rename planck chain
    planck_chain.setParamNames(['logAs','ns','H0','omegab','omegac'])#['omegab','omegac','H0','tau','ns','logAs','Aplanck'])
    lsst_chain.setParamNames(['logAs','ns','H0','omegab','omegac','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5'])
    
    chains = diff.chain()
    chains.chains = [planck_chain,lsst_chain]     
    chains.diff(feedback=False) # compute the difference chain
    # print(planck_chain.getMeans())
    # print(lsst_chain.getMeans()[:5])
    # print(chains.diff_chain.getMeans())
    
    print()
    
    ### Normalizing flow
    maf = flow.MAF(len(chains.params))
    maf.setup(feedback=False)
    maf.train(chains.diff_chain,batch_size=8192,feedback=False)
    nsigma,high,low = tension.flow_significance(
                        maf.target_dist,
                        maf.gauss_bijector,
                        len(chains.params)
                        )
    print(chain,nsigma)
print('done!')
