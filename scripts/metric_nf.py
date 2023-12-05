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

def compute_omega_b_m(pars):
    params   = pars[:,:6]
    h        = pars[:,2]
    omegabh2 = pars[:,0]
    omegach2 = pars[:,1]

    omeganh2 = (3.046/3)**(3/4)*0.06/94.1

    omegamh2 = omegabh2+omegach2+omeganh2

    omegab = omegabh2/h**2
    omegam = omegamh2/h**2

    return(omegab,omegam)

# Paths for chains
lsst_path    = '/home/grads/extra_data/evan/shifted_chains/'
planck_path  = '/home/grads/extra_data/evan/planck_chains/'

start = 0   # range to run results
stop  = 256 # the number of noise realizations
shift = -5   # the shift to consider
results = []

if shift==5:
    savename = 'nf_p5.txt' #'planck_lsst_nf_p5.txt'
if shift==0:
    savename = 'nf_0.txt' #'planck_lsst_nf_0.txt'
if shift==-5:
    savename = 'nf_m5.txt' #'planck_lsst_nf_m5.txt'

for i in range(start,stop):
    planck_chain = getdist.mcsamples.loadMCSamples('/home/grads/data/evan/cosmopower_emcee/notebooks/planck_emulated',no_cache=True)#planck_path+str(i))
    try:
        lsst_chain   = getdist.mcsamples.loadMCSamples(lsst_path+'shift_'+str(shift)+'_noise'+str(i),no_cache=True)
    except:
        print('{} not found')
        results.append(-1)
        continue
        
    samples = lsst_chain.samples
    omega_b,omega_c = compute_omega_b_c(samples)

    samples[:,3] = omega_b
    samples[:,4] = omega_c

    lsst_chain.setSamples(samples)
    
    #need to convert h to H0
    s = planck_chain.samples
    s[...,2] = 100*s[...,2] 
    planck_chain.setSamples(s)

    #rename planck chain
    planck_chain.setParamNames(['omegab','omegac','H0','tau','ns','logAs','Aplanck'])
    lsst_chain.setParamNames(['logAs','ns','H0','omegab','omegac','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5'])

    ##### START DEBUG #####
    # s = planck_chain.samples
    # omega_b,omega_m = compute_omega_b_m(s)
    # s[...,0] = omega_b
    # s[...,1] = omega_m
    # s[...,2] = 100*s[...,2] 
    # planck_chain.setSamples(s)

    # #rename planck chain
    # planck_chain.setParamNames(['omegab','omegac','H0','tau','ns','logAs','Aplanck'])
    # lsst_chain.setParamNames(['logAs','ns','H0','omegab','omegac','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5'])
    ##### END DEBUG #####

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
    results.append(nsigma)
    #print(planck_chain.getMeans(),lsst_chain.getMeans(),nsigma)
    if( i%10==0 ):
        np.savetxt(savename,np.array(results))
        print('Checkpointed!')

np.savetxt(savename,np.array(results))
print('done!')
