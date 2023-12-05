import getdist
import numpy as np
import scipy
from . import flow
from . import diff
from . import tension
import os
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '../../'))

def qudm(chain,update_chain,feedback=False):
    # input 2 chains
    # return tension n_sigma
    
    # Get means and covariance
    param_names_1 = chain.getParamNames().getRunningNames()
    param_names_2 = update_chain.getParamNames().getRunningNames()
    common_params = [param for param in param_names_1 if param in param_names_2]

    cov_A = chain.cov(common_params)
    mean_A = chain.mean(common_params)
    cov_AB = update_chain.cov(common_params)
    mean_AB = update_chain.mean(common_params)

    # difference tensors
    cov_diff = cov_A - cov_AB
    inv_cov_diff = np.linalg.inv(cov_diff)
    mean_diff = mean_A - mean_AB
    
    #find degrees of freedom
    rank = np.linalg.matrix_rank(cov_diff)

    # compute qudm
    qudm = mean_diff @ inv_cov_diff @ mean_diff

    pte = scipy.stats.chi2.cdf(qudm, rank)
    n_sigma = np.sqrt(2)*scipy.special.erfinv(pte)
    
    # print info if desired
    if( feedback ):
        print('dof = {}'.format(rank))
        print('Q_UDM = {:.5e}'.format(qudm))
        print('\nn_sigma = {}'.format(n_sigma))
        
    return n_sigma

def N_eff(chain,prior=None,prior_cov=None):
    N = len(chain.getParamNames().getRunningNames())
    post_cov = chain.cov(chain.getParamNames().getRunningNames())
    if( prior is not None ):
        pri_cov = prior.cov(chain.getParamNames().getRunningNames())
    elif( prior_cov is None ):
        raise ValueError('No prior provided!')
    else:
        pri_cov = prior_cov
    return N - np.trace(np.linalg.inv(pri_cov) @ post_cov)

def max_a_post(chain):
    # sometimes getdist will have the necessary information:
    try:
        best_fit = chain.getBestFit(max_posterior=True)
        lkl = best_fit.logLike
    # when it doesn't use scipy + emulation
    except:
        RuntimeError('Best fit not found in {}. \n\
                Pass likelihood to class instead.'.format(chain))
    return lkl
        
def prior_cov(chain=None,dictionary=None):
    try:
        n_params = len(chain.getParamNames().getRunningNames())
    except:
        n_params = len(dictionary.keys())
    cov_prior = np.zeros((n_params,n_params))
    for i in range(n_params):
        name = chain.getParamNames().getRunningNames()[i]
        # get prior limits
        try:
            amin = chain.ranges.lower[name]
        except:
            amin = dictionary[name][0]
        try:
            amax = chain.ranges.upper[name]
        except:
            amax = dictionary[name][1]

        # sample from prior
        #pr_samples.append(np.random.uniform(amin,amax,1000000))
        cov_prior[i,i] = ((amax-amin)**2)/12
        
    return cov_prior
            
def q_dmap(chain_a,chain_b,chain_ab,chain_prior=None,prior_dict=None,lkl_a=None,lkl_b=None,lkl_ab=None,feedback=False):
    # for data set a and b, 
    # chain_ab is the joint chain
    
    # start with N_eff
    if( chain_prior is None ):
        #print('No prior chain given. Generating samples assuming hard prior...')
        
        cov_prior_a = prior_cov(chain_a,prior_dict)
        cov_prior_b = prior_cov(chain_b,prior_dict)
        cov_prior_ab = prior_cov(chain_ab,prior_dict)
        
        neff_a = N_eff(chain_a,prior_cov=cov_prior_a)
        neff_b = N_eff(chain_a,prior_cov=cov_prior_b)
        neff_ab = N_eff(chain_a,prior_cov=cov_prior_ab)

    else:
        neff_a = N_eff(chain_a,prior=chain_prior)
        neff_b = N_eff(chain_b,prior=chain_prior)
        neff_ab = N_eff(chain_ab,prior=chain_prior)
        
    # now find maximum a posteriori
    try:
        map_a = max_a_post(chain_a)
        map_b = max_a_post(chain_b)
        map_ab = max_a_post(chain_ab)
    except:
        try:
            if feedback:
                print('Using given likelihoods.')
            map_a = lkl_a
            map_b = lkl_b
            map_ab = lkl_ab
        except:
            RuntimeError('No likelihoods found!')
    
    Q_DMAP = 2*(map_a+map_b-map_ab)
    if feedback:
        print('Q_DMAP = {:.5f}'.format(Q_DMAP))
    
    # get the degrees of freedom
    
    d = neff_a + neff_b - neff_ab
    if feedback:
        print('dof = {:.5f}'.format(d))
    
    # compute n_sigma
    pte = scipy.stats.chi2.cdf(Q_DMAP, d)
    n_sigma = np.sqrt(2)*scipy.special.erfinv(pte)
    return n_sigma

def eigentension(chain1,chain2,prior_lims=None,feedback=False):
    # get cov of chains
    param_names_1 = chain1.getParamNames().getRunningNames()
    param_names_2 = chain2.getParamNames().getRunningNames()
    common_params = [param for param in param_names_1 if param in param_names_2]

    cov_1 = chain1.cov(common_params)
    evals1,evecs1 = np.linalg.eigh(cov_1)
    
    # compute diagonal cov
    inv_evecs = np.linalg.inv(evecs1)
    d_cov = inv_evecs @ cov_1 @ evecs1
    
    # get prior cov assumming hard prior.
    idx = [param_names_1.index(param) for param in common_params]
    samples = chain1.samples
    # cov_prior = np.zeros((len(idx),len(idx)))

    vecs = samples[...,idx]
    pr_samples = []
    
    for i in range(len(common_params)):
        name = common_params[i]
        # get prior limits
        # try:
        #     amin = chain1.ranges.lower[name]
        # except:
        #     amin = prior_lims[name][0]
        # try:
        #     amax = chain1.ranges.upper[name]
        # except:
        #     amax = prior_lims[name][1]
        amin = prior_lims[name][0]
        amax = prior_lims[name][1]

        # sample from prior
        pr_samples.append(np.random.uniform(amin,amax,50000))
        #cov_prior[i,i] = ((amax-amin)**2)/12

    cov_prior = np.cov(pr_samples)

    #change to eigenbasis
    # t_pr_samples = pr_samples
    t_pr_samples = np.linalg.inv(evecs1) @ pr_samples
    transformed_vecs=np.transpose(np.linalg.inv(evecs1) @ np.transpose(vecs))

    dcov_prior=inv_evecs @ cov_prior @ evecs1
    # variances
    # for i in range(len(common_params)):
    #     cov_prior[i,i] = np.var(pr_samples[i])
        
    # get the ratio of variances
    r = np.zeros(len(dcov_prior))
    robust=[]

    for i in range(len(dcov_prior)):
        r[i] = dcov_prior[i,i]/d_cov[i,i]

    for i in range(len(r)):
        if( r[i]>100 ):
            robust.append(i)
            if feedback:
                print(f'{r[i]} is the ratio of the prior to the posterior variance in the {i}-th direction.')
                print('{} is robust!'.format(i))
     
    # make sure at least 2 robust eigenmodes so I can use normalizing flows
    if( len(robust)<2 ):
        add_vec = -1*np.sort(-1*r)[1]
        new_idx = np.where(r==add_vec)
        robust.append(new_idx[0][0])
        if feedback:
            print('adding the {} eigenvector to well-measured subspace'.format(new_idx[0]))
        
    idx = [param_names_2.index(param) for param in common_params]
    print(idx)
    s2 = chain2.samples[...,idx]
    s2_proj = np.transpose(np.linalg.inv(evecs1) @ np.transpose(s2))
    print(transformed_vecs[0,robust])
    print(s2_proj[0,robust])

    well_measured_1 = s2_proj[...,robust]
    well_measured_2 = transformed_vecs[...,robust]
    
    n_samples = max(len(well_measured_1),len(well_measured_2))
    
    # now turn them into chains and use NF
    chain1 = getdist.mcsamples.MCSamples(samples=well_measured_1,names=['e'+str(i) for i in robust],label='chain1')
    chain2 = getdist.mcsamples.MCSamples(samples=well_measured_2,names=['e'+str(i) for i in robust],label='chain2')

    chain1.saveAsText('eigen_chain1')
    chain2.saveAsText('eigen_chain2')
    
    # difference chain
    chains = diff.chain()
    chains.chains = [chain1,chain2]
    chains.diff(feedback=feedback)
    
    # MAF
    maf = flow.MAF(len(chains.params))
    maf.setup(feedback=feedback)
    maf.train(chains.diff_chain,batch_size=int(n_samples/100),feedback=feedback)
    
    nsigma,high,low = tension.flow_significance(
                        maf.target_dist,
                        maf.gauss_bijector,
                        len(chains.params)
                        )
    if feedback:
        print(r"n_sigma = {:.5f} +{:.5f}/-{:.5f}".format(nsigma,high-nsigma,nsigma-low))
        
    return nsigma,high,low