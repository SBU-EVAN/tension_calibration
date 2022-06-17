#
# compute parameter difference distributions. 
# return hash table of data and params
#
#     - numpy.loadtxt
#     - emcee h5 backend
#
# Much of the implementation follows Marco Raveri's Tensiometer
#

import numpy as np

try:
    import emcee
except ImportError:
    print('Warning: Emcee not installed')
try:
    import getdist
    from getdist import MCSamples, WeightedSamples
except:
    print('Warning: Getdist not installed')

class chain:
    def __init__(self,file=None,chains=None,diff_chain=None, params=None):
        # constructor
        # file: string with path to file to read
        '''
        if( file != None ):
            self.file = file
        else:
            print('Warning: File name not specified')
        if( backend != None ):
            self.backend = backend
        else:
            print('Warning: No backend specified')
        '''
        self.chains = chains
        self.diff_chain = diff_chain
        self.params = params
    
    def h5_reader(self,file=None):
        # uses emcee HDF backend to read .h5 file
        try:
            reader = emcee.backends.HDFBackend(file, read_only=True)
        except:
            raise TypeError('Must specify file path')
        chain = reader.get_chain(flat=True)
        return chain
    
    def getdist_reader(self,file=None):
        # getdist MCSamples can be saved in multiple files all sharing a root.
        # read all and make them into a single chain, return one chain.
        try:
            chain = getdist.mcsamples.loadMCSamples(file_root=file, no_cache=True)
        except:
            raise TypeError('Must specify file path')
        if( chain.getSeparateChains() != [] ):
            samples_list = chain.getSeparateChains()
            chain.chains = samples_list
            chain.makeSingle()
        return chain
    
    def diff(self,boost=1):
        # self.chains = list/array of length 2
        # boost is a way to generate more samples if needed.
        # boost should be 1 unless testing
        #
        # returns: chain of type MCSamples()
        chains = self.chains
        # check chains is valid
        if( type(chains) == None ):
            raise TypeError("Chains must be list or array.")
        if( len(chains) != 2 ):
            raise AttributeError("Chain list must have length 2")
            
        # separate the chains
        chain0 = chains[0].samples
        chain1 = chains[1].samples
        w_chain0 = chains[0].weights
        w_chain1 = chains[1].weights
        ll_chain0 = chains[0].loglikes
        ll_chain1 = chains[1].loglikes
    
        # get the names
        # sometimes people put in the same parameters with a different name
        # print the running params so users can verify the params are recognized
        names_0 = chains[0].getParamNames().getRunningNames()
        names_1 = chains[1].getParamNames().getRunningNames()
        print('\nParams in first chain:')
        print(names_0)
        print('\nParams in second chain:')
        print(names_1)
        
        # get common params
        common_params = [param for param in names_0 if param in names_1]
        print('\n Common params found:')
        print(common_params)
        
        # find the indices of the common parameters, accounts for different order
        idx0 = [names_0.index(param) for param in common_params]
        idx1 = [names_1.index(param) for param in common_params]
        assert len(idx0)==len(idx1)
    
        # ensure first chain is longer than the second.
        # Need to keep track if I flipped the data so I get the signs right 
        # (although in principle it doesn't matter, Its better for everyones results 
        # to look the same even if they import chains in different orders)
        flip=False
        if( len(chain0) < len(chain1) ):
            chain0,chain1 = chain1,chain0
            w_chain0,w_chain1 = w_chain1,w_chain0
            ll_chain0,ll_chain1 = ll_chain1,ll_chain0
            idx0,idx1 = idx1,idx0
            flip=True

        N0 = len(chain0)
        N1 = len(chain1)
        print('\nN1 = {}'.format(N0))
        print('N2 = {}'.format(N1))

        # set up parameter diff arrays
        diff = np.zeros((N0*boost,len(idx0)),dtype=np.float32)
        weights = np.zeros(N0*boost,dtype=np.float32)
        loglikes = np.zeros(N0*boost,dtype=np.float32)

        # The boost can be used to increase the number of samples you have
        # by wrapping the data multiple times. It should only be changed for testing.
        # Full chains should be computed with enough samples without the boost.
        
        for i in range(boost):
            # find the range of indices to use for chain 2
            lower = int((i/boost)*N0)
            upper = lower+N0

            # compute stuff
            if flip==True:
                diff[i*N0:(i+1)*N0] =\
                    -chain0[:N0,idx0] + np.take(chain1[:,idx1], \
                    range(lower,upper), axis=0, mode='wrap')
            else:
                diff[i*N0:(i+1)*N0] = \
                    chain0[:N0,idx0] - np.take(chain1[:,idx1],\
                    range(lower,upper), axis=0, mode='wrap')
               
            weights[i*N0:(i+1)*N0] = w_chain0*np.take(w_chain1, range(lower,upper), mode='wrap')
            if(ll_chain0 is not None and ll_chain1 is not None):
                loglikes[i*N0:(i+1)*N0] = ll_chain0+np.take(ll_chain1, range(lower,upper), mode='wrap')

        min_weight_ratio = min(chains[0].min_weight_ratio,
                                   chains[1].min_weight_ratio)
        # make the weighted samples
        diff_samples = WeightedSamples(ignore_rows=0,
                                       samples=diff,
                                       weights=weights, loglikes=loglikes,
                                       name_tag=' ', label=' ',
                                       min_weight_ratio=min_weight_ratio)
        
        # turn back into MCSamples type
        chain = MCSamples(names=common_params,labels=common_params)
        chain.chains = [diff_samples]
        chain.makeSingle()

        self.diff_chain = chain
        self.params = common_params
        return
        