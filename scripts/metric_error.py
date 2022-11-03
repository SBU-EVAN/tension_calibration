<<<<<<< HEAD
#!/usr/bin/env pythonr
=======
# In[33]:

>>>>>>> 3c82b715d6e59c9c7c02b61268f04b833ab1c65d

import sys
import os
import numpy as np
import getdist
import matplotlib.pyplot as plt
<<<<<<< HEAD
=======
#get_ipython().run_line_magic('matplotlib', 'inline')
>>>>>>> 3c82b715d6e59c9c7c02b61268f04b833ab1c65d

sys.path.append(os.path.join(os.path.dirname("__file__"), '../'))
from metrics import diff
from metrics import flow
from metrics import tension
from metrics.parameter_metrics import *
from metrics import utilities
from emulators import lsst
from emulators import cosmopower
<<<<<<< HEAD
import pybobyqa


# Paths for chains
lsst_path    = '/home/grads/extra_data/evan/lsst_shifts/sigma8/'
planck_path  = '/home/grads/extra_data/evan/planck_chains/'
joint_path = '/home/grads/extra_data/evan/joint_chains/'
savename = 'sigma8_-3_results.txt'

#######
# Open the emulators, setup functions, intial pos of optimizer
#######

lsst_emulator = lsst.lsst_emulator()
cosmopower_emulator = cosmopower.cosmopower()

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

### for GOF degradation need likelihoods and posterior

def planck_log_prob(theta):
    theta = theta.astype('float32')
    p = cosmopower_emulator.tf_planck.posterior(np.array([theta],np.float32)).numpy()[0]
    if( p == -1*np.inf ):
        p = -1e32
    return -1 * p

def lsst_log_prob(theta):
    p = lsst_emulator.log_prob(theta)
    if( p == -1*np.inf ):
        p = -1e32
    return -1 * p

def joint_log_prob(theta):
    planck_idxs = [3,4,2,29,1,0,30]
    theta_lsst = theta[:29]
    theta_planck = np.array(theta)[planck_idxs]
    theta_planck[2] = theta_planck[2]/100
    lsst_prob = lsst_emulator.log_prob(theta_lsst)
    planck_prob = cosmopower_emulator.tf_planck.posterior(np.array([theta_planck],dtype=np.float32)).numpy()[0]
    p = ( lsst_prob + planck_prob )
    if( p == -1*np.inf ):
        p = -1e32
    return -1 * p

### bounds for the optimizer
planck_bounds  = [[0.001,0.005,0.55,0.01,0.9,1.61,0.95],[0.04,0.99,0.91,0.8,1.1,3.91,1.05]]

lsst_bounds = np.array([[1.61,0.9,55,0.001,0.005,
                            -1,-1,-1,-1,-1,
                            0,-1,
                            -1,-1,-1,-1,-1,
                            0,0,0,0,0,
                            -1,-1,-1,-1,-1,
                            -1,-1
                       ],
                       [3.91,1.1,91,0.04,0.99,
                            1,1,1,1,1,
                            1,1,
                            1,1,1,1,1,
                            10,10,10,10,10,
                            1,1,1,1,1,
                            1,1
                            ]])

joint_bounds = np.array([[1.61,0.9,55,0.001,0.005,
                            -1,-1,-1,-1,-1,
                            0,-1,
                            -1,-1,-1,-1,-1,
                            0,0,0,0,0,
                            -1,-1,-1,-1,-1,
                            -1,-1,0,0
                       ],
                       [3.91,1.1,91,0.04,0.99,
                            1,1,1,1,1,
                            1,1,
                            1,1,1,1,1,
                            10,10,10,10,10,
                            1,1,1,1,1,
                            1,1,1,10
                            ]])

### Change as needed
overwrite = True  # If false only adds to results and reruns those with error codes

_nf        = True  # If true runs this metric
_q_udm     = False
_q_dmap    = False
_eigenmode = True

print('\nNormalizing Flow : {}'.format(_nf))
print('Q_UDM            : {}'.format(_q_udm))
print('Q_DMAP           : {}'.format(_q_dmap))
print('Eigentension     : {}\n'.format(_eigenmode))

start = 0          # range to run results
stop = 1000
####################

####################
# FOR QDMAP DEBUG ONLY
arr_a = []
arr_b = []
arr_ab = []
####################

#open planck and lsst noise realizations
planck_dvs = np.loadtxt('/home/grads/data/evan/tension_calibration/planck_noise.txt')
lsst_dvs   = np.loadtxt('/home/grads/data/evan/tension_calibration/shift_sigma8_-3.txt')

### open results or create new results
if overwrite:
    results = []                           
else:
    try:
        results = np.loadtxt(savename).tolist()
    except:
        print('no results.txt found!')
        results = []

### Now run the metrics
for i in range(start,stop):
    #print('\r'+str(i),end='')
    try:
        r = results[i]
        append = False # flag that determines whether to use append or index
        #print(r)
        if( (np.array(r[1:])>=0).all() and len(r)>1 ):
            continue
        else:
            r = [i]
    except:
        r = [i] # holds only this iteration results
        #print('need to run!')
        append = True # flag that determines whether to use append or index
    try:
        chain1 = getdist.mcsamples.loadMCSamples(planck_path+str(i))
        chain2 = getdist.mcsamples.loadMCSamples(lsst_path+'lsst_sigma8_-3_'+str(i))
        #chain3 = getdist.mcsamples.loadMCSamples(joint_path+'joint_'+str(i))
        chain1.removeBurnFraction(0.3) # forgot :(
        chain1.weighted_thin(10)       #forgot
    except:
        print('\r'+str(i)+' is missing!!!')
        r = [i]
        for j in range(4):
            r.append(-3) # set all metrics to -3
        if( append ):
            results.append(r)
        else:
            results[i] = r
        continue        # skip rest of loop
        
    s = chain1.samples
    s[...,2] = 100*s[...,2]
    chain1.setSamples(s)
        
    chains = diff.chain()
    chains.chains = [chain1,chain2]     
    chains.diff(feedback=False) # compute the difference chain
    
    print()
    
    ### Normalizing flow
    if _nf:     
        maf = flow.MAF(len(chains.params))
        maf.setup(feedback=False)
        maf.train(chains.diff_chain,batch_size=10000,feedback=False)
        nsigma,high,low = tension.flow_significance(
                            maf.target_dist,
                            maf.gauss_bijector,
                            len(chains.params)
                            )
        if( np.isnan(nsigma) or nsigma==np.inf ):
            r.append(-2)
        else:
            r.append(nsigma)  
    else:
        r.append(-1)
        
    ### Parameter Update
    if _q_udm:
        nsigma = qudm(chain2,chain3,feedback=False)
        if( np.isnan(nsigma) or nsigma==np.inf ):
            r.append(-2)
        else:
            r.append(nsigma)
    else:
        r.append(-1)
        
    ### GOF degredation
    if _q_dmap:
        # set noise realizations
        #print('\nstart_opt')
        cosmopower_emulator.tf_planck.X_data = planck_dvs[i].astype(np.float32)
        lsst_emulator.data_model.dv_obs = lsst_dvs[i]
        
        # PLANCK
        planck_init = chain1.getMeans()
        planck_init[2] = planck_init[2]/100
        planck_init[-1] = 1.0
        map_a = pybobyqa.solve(planck_log_prob,planck_init,rhobeg=0.1,rhoend=1e-12,bounds=planck_bounds,scaling_within_bounds=True,maxfun=1000)
        lkl_a = cosmopower_emulator.tf_planck.get_loglkl(np.array([map_a.x],dtype=np.float32)).numpy()[0]
        #print('map_a')
        #print(map_a)

        # LSST
        _lsst_init = chain2.getMeans()
        #print(_lsst_init)
        lsst_init = np.array([_lsst_init[0],_lsst_init[1],_lsst_init[2],_lsst_init[3],_lsst_init[4],
          0., 0., 0., 0., 0.,
          0.501, 0.,
          0., 0., 0., 0., 0.,
          1.241, 1.361, 1.471, 1.601, 1.761,
          0., 0., 0., 0., 0.,
          0., 0.])
        #print(lsst_init)
        map_b = pybobyqa.solve(lsst_log_prob,lsst_init,rhobeg=0.1,rhoend=1e-12,bounds=lsst_bounds,scaling_within_bounds=True,maxfun=1000)
        lkl_b = lsst_emulator.log_lkl(map_b.x)
        #print('map_b')
        #print(map_b)

        # JOINT POSTERIOR
        _joint_init = chain3.getMeans()
        joint_init = np.array([_joint_init[2],_joint_init[3],_joint_init[4],_joint_init[0],_joint_init[1],
                      0., 0., 0., 0., 0.,
                      0.501, 0.,
                      0., 0., 0., 0., 0.,
                      1.241, 1.361, 1.471, 1.601, 1.761,
                      0., 0., 0., 0., 0.,
                      0., 0.,_joint_init[5],_joint_init[6]])
        #print(joint_init)
        map_ab = pybobyqa.solve(joint_log_prob,joint_init,rhobeg=0.1,rhoend=1e-12,bounds=joint_bounds,scaling_within_bounds=True,maxfun=1000)
        
        planck_idxs = [3,4,2,29,1,0,30]
        map_ab_lsst = map_ab.x[:29]
        map_ab_planck =  map_ab.x[planck_idxs]
        map_ab_planck[2] = map_ab_planck[2]/100
        lkl_ab = lsst_emulator.log_lkl(map_ab_lsst)+cosmopower_emulator.tf_planck.get_loglkl(np.array([map_ab_planck],dtype=np.float32)).numpy()[0]
        #print('map_ab')
        #print(map_ab)
        
        nsigma = q_dmap(chain1,chain2,chain3,prior_dict=priors,lkl_a=lkl_a,lkl_b=lkl_b,lkl_ab=lkl_ab)
        
        if( np.isnan(nsigma) or nsigma==np.inf ):
            r.append(-2)
        else:
            r.append(nsigma)
            ######
            # QDMAP ONLY
            #_map_a = list(map_a.x)
            #_map_a.append(map_a.f)
            #_map_a.append(lkl_a)
            #arr_a.append(_map_a)

            #_map_b = list(map_b.x)
            #_map_b.append(map_b.f)
            #_map_b.append(lkl_b)
            #arr_b.append(_map_b)

            #_map_ab = list(map_ab.x)
            #_map_b.append(map_ab.f)
            #_map_b.append(lkl_ab)
            #arr_ab.append(_map_ab)
            ######
    else:
        r.append(-1)
    
        
    ### eigentension
    if _eigenmode:
        nsigma,high,low = eigentension(chain1,chain2,priors,feedback=False)
        if( np.isnan(nsigma) or nsigma==np.inf ):
            r.append(-2)
        else:
            r.append(nsigma)
    else:
        r.append(-1)
    if( append ):
        results.append(r)
    else:
        results[i] = r
    #print(r)
    if( i%10==0 ):
        np.savetxt(savename,results)
        print('Checkpointed!')

np.savetxt(savename,results)
#arr_a = np.array(arr_a)
#arr_b = np.array(arr_b)
#arr_ab = np.array(arr_ab)

#np.savetxt('qdmap_a.txt',arr_a)
#np.savetxt('qdmap_b.txt',arr_b)
#np.savetxt('qdmap_ab.txt',arr_ab)

# now count how many good and bad runs
results = np.loadtxt(savename) # load our results
good_results = []                   # the data that worked
bad_results  = []                   # the data that did not, this now has error codes

for i in range(len(results)):
    metrics = results[i,1:]
    if( (metrics<=0.).all() ):
        bad_results.append(results[i,0])
    else:
        good_results.append(metrics)
print('N Good Runs = {}'.format(len(good_results)))
print('N Bad Runs = {}'.format(len(bad_results)))
np.savetxt('bad_results.txt',bad_results)

#####
# Plotting. uncomment to plot. also can use notebook
#####
'''
good_results_T = np.transpose(good_results)
chain1 = getdist.MCSamples(samples=np.array(good_results_T[0]),names=['nf'],labels=['NF'])
chain2 = getdist.MCSamples(samples=np.array(good_results_T[1]),names=['Q_UDM'],labels=['Q_{UDM}'])
chain3 = getdist.MCSamples(samples=np.array(good_results_T[2]),names=['Q_DMAP'],labels=['Q_{DMAP}'])
chain4 = getdist.MCSamples(samples=np.array(good_results_T[3]),names=['eigen'],labels=['Eigentension'])

#limits and means
mean_nf = chain1.getMeans()
lims_nf = chain1.twoTailLimits(0, 0.67)
lims_nf2 = chain1.twoTailLimits(0,0.95)
lims_nf3 = chain1.twoTailLimits(0,0.997)

mean_qudm = chain2.getMeans()
lims_qudm = chain2.twoTailLimits(0, 0.67)
lims_qudm2 = chain2.twoTailLimits(0,0.95)
lims_qudm3 = chain2.twoTailLimits(0,0.997)

mean_qdmap = chain3.getMeans()
lims_qdmap = chain3.twoTailLimits(0, 0.67)
lims_qdmap2 = chain3.twoTailLimits(0,0.95)
lims_qdmap3 = chain3.twoTailLimits(0,0.997)

mean_eigen = chain4.getMeans()
lims_eigen = chain4.twoTailLimits(0, 0.67)
lims_eigen2 = chain4.twoTailLimits(0, 0.95)
lims_eigen3 = chain4.twoTailLimits(0, 0.997)

x = np.arange(1,5,1)
mean = np.mean([mean_nf,mean_qudm,mean_qdmap,mean_eigen])

plt.plot(x[0],mean_nf,'b',marker='o',lw=0,label='Param Diff + NF')
plt.plot([x[0],x[0]],[lims_nf[0],lims_nf[1]],c='b')
#plt.plot([x[0],x[0]],[lims_nf2[0],lims_nf2[1]],'b--')
#plt.plot([x[0],x[0]],[lims_nf3[0],lims_nf3[1]],'b:')

plt.plot(x[1],mean_qudm,'g',marker='o',lw=0,label='$Q_{UDM}$')
plt.plot([x[1],x[1]],[lims_qudm[0],lims_qudm[1]],c='g')
#plt.plot([x[1],x[1]],[lims_qudm2[0],lims_qudm2[1]],'g--')
#plt.plot([x[1],x[1]],[lims_qudm3[0],lims_qudm3[1]],'g:')

plt.plot(x[2],mean_qdmap,'c',marker='o',lw=0,label='$Q_{DMAP}$')
plt.plot([x[2],x[2]],[lims_qdmap[0],lims_qdmap[1]],c='c')
#plt.plot([x[2],x[2]],[lims_qdmap2[0],lims_qdmap2[1]],'c--')
#plt.plot([x[2],x[2]],[lims_qdmap3[0],lims_qdmap3[1]],'c:')

plt.plot(x[3],mean_eigen,marker='o',c='r',lw=0,label='Eigentension + NF')
plt.plot([x[3],x[3]],[lims_eigen[0],lims_eigen[1]],'r')
#plt.plot([x[3],x[3]],[lims_eigen2[0],lims_eigen2[1]],'r--')
#plt.plot([x[3],x[3]],[lims_eigen3[0],lims_eigen3[1]],'r:')

plt.plot(0,0,ls='-',c='k',label='CL=0.67')
#plt.plot(0,0,ls='--',c='k',label='CL=0.95')
#plt.plot(0,0,ls=':',c='k',label='CL=0.997')

plt.plot([0.9,6.5],[mean,mean],'k-.',label='mean n_sigma')
plt.ylabel('$n_\sigma$')
plt.xlim([0.9, 6.5])
plt.legend()
plt.savefig('metrics.pdf')  


# In[58]:


plt.hist(chain1.samples,bins=np.arange(0,4,0.25),density=True)
plt.plot([1.07220,1.07220],[0,1.0]) # 1.07220 is fiducial nf result
plt.xlabel('$n_\sigma$ NF')
plt.savefig('nf_hist.pdf')


# In[61]:


plt.hist(chain4.samples,bins=np.arange(0,4,0.25),density=True)
plt.plot([1.45829,1.45829],[0,1.5]) # 1.45829 is fiducial eigenmode result
plt.xlabel('$n_\sigma$ Eigen')
plt.savefig('eigen_hist.pdf')
'''
print('done!')
=======


# In[6]:


lsst_path = '/home/grads/extra_data/evan/lsst_chains/'
planck_path = '/home/grads/extra_data/evan/planck_chains/'


# In[25]:


# mcmc preprocessing params
burnin=5000
thin=10

# metric arrays
nf    = []
q_udm = []
qdmap = []
eigen = []

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

for i in range(2,950):
    print(str(i)+'/950')
    chain1 = chains.getdist_reader(planck_path+str(i))
    chain2 = chains.getdist_reader(lsst_path+'lsst_'+str(i))
    
    # some preprocessing
    samples1 = chain1.samples[burnin::thin]
    weights1 = chain1.weights[burnin::thin]
    samples1[:,2] = 100*samples1[:,2] # wtf
    chain1.setSamples(samples1,weights=weights1)
    
    samples2 = chain2.samples[burnin::thin]
    weights2 = chain2.weights[burnin::thin]
    chain2.setSamples(samples2,weights=weights2)
    
    #now go
    chains.chains = [chain1,chain2]
    chains.diff()
    
    # MAF
    maf = flow.MAF(len(chains.params))
    maf.setup(feedback=False)
    maf.train(chains.diff_chain,batch_size=5000,feedback=False)
    nsigma,high,low = tension.flow_significance(
                        maf.target_dist,
                        maf.gauss_bijector,
                        len(chains.params)
                        )
    print(r"n_sigma = {:.5f} +{:.5f}/-{:.5f}".format(nsigma,high-nsigma,nsigma-low))
    nf.append(nsigma)
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
    '''
    # eigentension
    nsigma,high,low = eigentension(chain2,chain1,priors)
    eigen.append(nsigma)


# In[30]:


print(nf)
print(q_udm)
print(qdmap)
print(eigen)


# In[31]:
np.savetxt('nf.txt',nf)
np.savetxt('eigen.txt',eigen)

mean_nf = np.mean(nf)
std_nf = np.std(nf)
'''
mean_qudm = np.mean(q_udm)
std_qudm = np.mean(q_udm)

mean_qdmap = np.mean(qdmap)
std_qdmap = np.mean(qdmap)
'''
mean_eigen = np.mean(eigen)
std_eigen = np.mean(eigen)


# In[34]:

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
>>>>>>> 3c82b715d6e59c9c7c02b61268f04b833ab1c65d
