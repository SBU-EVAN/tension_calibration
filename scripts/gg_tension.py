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
from emulators import lsst
from emulators import cosmopower

chain_path = '/home/grads/data/KunHao/GeoGrowth/gg_split_tension/chains/'
chain_list = sorted(os.listdir(chain_path)) #list of all chains

chains_needed = np.array([72, 74, 76, 78, 80, 82, 83, 84, 85, 86, 87, 88],dtype=str) # only need these chains
completed_pairs = [] #Only need lower triangular part

# mcmc preprocessing params
#burnin=5000
#thin=10

# metric arrays
nf    = []
q_udm = []
#qdmap = []
#eigen = []

# priors for eigentension
priors = {'omegabh2':  [0.001, 0.04],
          'omegach2':  [0.005, 0.99],
          'tau':     [0.01,  0.8],
          'ns':      [0.9,   1.1],
          'logA':    [1.61,  3.91],
          'Aplanck': [1.0,   0.01],
          'H0':      [55, 91],
         }

# instantiate diff
chains = diff.chain()

for file1 in chain_list:
    for file2 in chain_list:
        # check if this is a chain we want
        if( 'EXAMPLE_MCMC' not in file1 or 'EXAMPLE_MCMC' not in file2 ):
            continue
        # now check if it is a chain in the needed list
        if( file1[12:14] not in chains_needed or file2[12:14] not in chains_needed ):
            continue
        #:)
        if( '.txt' not in file1 or '.txt' not in file2 ):
            continue
        # make sure don't compute duplicate pairs
        if( [file1[:-4],file2[:-4]] in completed_pairs or [file2[:-4],file1[:-4]] in completed_pairs ):
            continue
        # and don't need tension between a chain and itself
        if( file1==file2 ):
            continue

        chain1 = chains.getdist_reader(chain_path+file1[:-4])
        chain2 = chains.getdist_reader(chain_path+file2[:-4])
        
        # some preprocessing
        name1 = chain1.getParamNames().list()
        name2 = chain2.getParamNames().list()

        params = ['omegab','omegam','As','ns','H0','sigma8_split']
        idx1 = [name1.index(param) for param in params]
        idx2 = [name2.index(param) for param in params]

        samples1 = chain1.samples[...,idx1]
        samples2 = chain2.samples[...,idx2]

        weight1 = chain1.weights
        weight2 = chain2.weights

        chain1 = getdist.MCSamples(samples=samples1,weights=weight1,names=params,labels=params)
        chain2 = getdist.MCSamples(samples=samples2,weights=weight2,names=params,labels=params)
        
        #now go
        chains.chains = [chain1,chain2]
        chains.diff()
                
        # MAF
        maf = flow.MAF(len(chains.params))
        maf.setup(feedback=False)
        maf.train(chains.diff_chain,batch_size=2000,feedback=False)
        nsigma,high,low = tension.flow_significance(
                            maf.target_dist,
                            maf.gauss_bijector,
                            len(chains.params)
                            )
        print(r"n_sigma = {:.5f} +{:.5f}/-{:.5f}".format(nsigma,high-nsigma,nsigma-low))
        nf.append(nsigma)

        completed_pairs.append([file1[:-4],file2[:-4]]) # dont need to do any of them twice :)

        '''
        # param update
        chain3 = chains.getdist_reader(lsst_planck_path)
        chain3.setParamNames(['omegab', 'omegac', 'logA', 'ns', 'H0', 'h', 'tau', 'Aplanck'])
        qudm_estimate = qudm(chain2,chain3,feedback=False)
        q_udm.append(qudm_estimate)

        # gof loss
        likea=0
        for param in chain1.getLikeStats().list():
            likea += chain1.getLikeStats().parWithName(param).bestfit_sample

        likeb=0
        for param in chain2.getLikeStats().list():
            likeb += chain2.getLikeStats().parWithName(param).bestfit_sample

        likeab=0
        for param in chain3.getLikeStats().list():
            likeab += chain3.getLikeStats().parWithName(param).bestfit_sample

        n_sigma = q_dmap(chain2,chain1,chain3,prior_dict=priors,lkl_a=likea,lkl_b=likeb,lkl_ab=likeab)
        qdmap.append(n_sigma)
        
        # eigentension
        nsigma,high,low = eigentension(chain2,chain1,priors)
        eigen.append(nsigma)
        '''

print(nf)
print(q_udm)
#print(qdmap)
#print(eigen)
np.savetxt('chain_pairs.txt',completed_pairs,fmt="%s")

# save
np.savetxt('gg_nf.txt',nf)
#np.savetxt('eigen.txt',eigen)

mean_nf = np.mean(nf)
std_nf = np.std(nf)
'''
mean_qudm = np.mean(q_udm)
std_qudm = np.mean(q_udm)

mean_qdmap = np.mean(qdmap)
std_qdmap = np.mean(qdmap)

mean_eigen = np.mean(eigen)
std_eigen = np.mean(eigen)
'''

'''
x = np.arange(1,5,1)
y = [mean_nf,mean_qudm,mean_qdmap,mean_eigen]
yerr = [std_nf,std_qudm,std_qdmap,std_eigen]
mean = np.mean([mean_nf,mean_qudm,mean_qdmap,mean_eigen])
plt.errorbar(x[0],mean_nf,yerr=std_nf,lw=0,elinewidth=2,marker='o',markersize=4,label='Param Diff + NF')
#plt.errorbar(x[1],mean_qudm,yerr=std_qudm,lw=0,elinewidth=2,marker='o',markersize=4,label='Param Diff Update')
#plt.errorbar(x[2],mean_qdmap,yerr=std_qdmap,lw=0,elinewidth=2,marker='o',markersize=4,label='Goodness of Fit Loss')
plt.errorbar(x[3],mean_eigen,yerr=std_eigen,lw=0,elinewidth=2,marker='o',markersize=4,label='Eigentension + NF')
plt.plot([0.9,6.5],[mean,mean],'k--',label='mean n_sigma')
plt.ylabel('$n_\sigma$')
plt.xlim([0.9, 6.5])
plt.legend()
plt.show()
'''
