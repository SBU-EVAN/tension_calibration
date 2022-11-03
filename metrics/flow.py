#
# Code for Normalizing flow shift estimation
# follows Marco Raveri's "tensiometer"
#

import numpy as np
import matplotlib.pyplot as plt
import getdist
from getdist import plots, MCSamples, WeightedSamples

# Now normalizing flow
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tf.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import Callback

from numpy import linalg
import scipy

### tensorflow training callbacks
class Callback(tfk.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self._loss = []
        self._epoch = []
        self.n_epochs = self.params['epochs']
        print('[                    ] Training... ',end="")
        
    def on_epoch_begin(self, epoch, logs=None):
        progress = int(epoch/self.n_epochs*20)
        ret = '\r['
        for i in range(progress):
            ret += '#'
        for i in range(20-progress):
            ret += ' '
        print(ret+'] Training... (epoch {}/{})'.format(epoch,self.n_epochs),end="")

    def on_epoch_end(self, epoch, logs=None):
        self._loss.append(logs['loss'])
        self._epoch.append(epoch)

    def on_train_end(self, logs=None):
        print('\r'+'[####################] Completed!                          ')
        fig,ax1 = plt.subplots(1,1)
        
        ax1.set_title('loss vs. epoch')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.plot(self._epoch,self._loss)
        
class No_Plot_Callback(tfk.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.n_epochs = self.params['epochs']
        print('[                    ] Training... ',end="")
        
    def on_epoch_begin(self, epoch, logs=None):
        progress = int(epoch/self.n_epochs*20)
        ret = '\r['
        for i in range(progress):
            ret += '#'
        for i in range(20-progress):
            ret += ' '
        print(ret+'] Training... (epoch {}/{})'.format(epoch,self.n_epochs),end="")

    def on_train_end(self, logs=None):
        print('\r'+'[####################] Completed!                             ')
        
class MAF:
    def __init__(self,
                 n_params,
                 hidden_units=None,
                 activation=None,
                 n_maf=None, 
                 permute=True, 
                 bijectors=None, 
                 base_dist=None, 
                 target_dist=None, 
                 gauss_bijector=None
                ):
                
        # sensible defaults
        if( hidden_units is None ):
            hidden_units = [2*n_params,2*n_params]
        if( activation is None ):
            activation = tf.math.asinh
        if( n_maf is None ):
            n_maf = 2*n_params
        if( base_dist is None ):
            base_dist =  tfd.MultivariateNormalDiag(
                            loc=np.zeros(n_params,dtype=np.float32), 
                            scale_diag=np.ones(n_params,dtype=np.float32))
        
        self.hidden_units = hidden_units
        self.activation   = activation
        self.n_maf        = n_maf
        self.n_params     = n_params
        self.permute      = permute
        self.base_dist    = base_dist
   
    def pregauss(self,chain,data):
        covmat = chain.cov().astype(np.float32)
        mean = chain.getMeans().astype(np.float32)

        # TriL means the cov matrix is lower triangular. Inverse is easy to compute that way
        # the cholesky factorization takes a positive definite hermitian matrix M 
        #(like the covmat) to LL^T with L lower triangluar
        gauss_approx = tfd.MultivariateNormalTriL(loc=mean,scale_tril=tf.linalg.cholesky(covmat))
        bijector = gauss_approx.bijector

        # now map the data
        new_data = bijector.inverse(data.astype(np.float32))
        return new_data,bijector

    def train(self,data=None,batch_size=5000,n_epochs=100,feedback=False,val_split=0.1,learning_rate=0.01):
        if( data is None ):
            raise TypeError('Must specify data as MCSamples type')
            
        # stack data
        _data = []
        dim = 0
        for key in data.getParamNames().list():
            nsamples=len(data[key])
            _data.append(data[key])
            dim += 1

        xdata = np.stack(_data, axis=-1)

        x_data,bij = self.pregauss(data,xdata)

        #create data set with weights.
        weights = data.weights.astype(np.float32)

        ## NN setup
        target_distribution = tfd.TransformedDistribution(
            distribution=self.base_dist,
            bijector=tfb.Chain(self.bijectors)) 

        # Construct model.
        x_ = tfk.Input(shape=(dim,), dtype=tf.float32)
        log_prob_ = target_distribution.log_prob(x_)
        model = tfk.Model(x_, log_prob_)

        model.compile(optimizer=tf.optimizers.Adam(learning_rate),
                      loss=lambda _, log_prob: -log_prob) 
        if(feedback):
            print('---   Model info   ---')
            print(" - N samples = {}".format(nsamples))
            if weights.all()==weights[0]:
                print(" - Uniform weights = {}".format(weights[0]))
            else:
                print(" - Non-uniform weights")
            print(" - Pre-Gaussian Map = True\n")
            print(" - Validation split = {}".format(val_split))
            print(' - Number MAFS = {} '.format(int(len(self.bijectors)/2)))
            print(' - Trainable parameters = {} \n'.format(model.count_params()))

        # now perform the fit
        if(feedback):
            model.fit(x=x_data,
                      y=np.zeros((nsamples, dim),dtype=np.float32),
                      batch_size=batch_size,
                      epochs=n_epochs,
                      steps_per_epoch=int((1-val_split)*nsamples/batch_size),
                      validation_split=val_split,
                      shuffle=True,
                      verbose=False,
                      callbacks=[Callback(),tfk.callbacks.ReduceLROnPlateau()]) 
        if(not feedback):
            model.fit(x=x_data,
                      y=np.zeros((nsamples, dim),dtype=np.float32),
                      batch_size=batch_size,
                      epochs=n_epochs,
                      steps_per_epoch=int((1-val_split)*nsamples/batch_size), 
                      validation_split=val_split,
                      shuffle=True,
                      verbose=False,
                      callbacks=[No_Plot_Callback(),tfk.callbacks.ReduceLROnPlateau()])

        self.target_dist = target_distribution
        self.gauss_bijector = bij
        #return(target_distribution,bij)

    def setup(self,feedback=False):
        # Set up bijector MADE
        if(feedback):
            print('---   MADE info   ---')
            print(' - Hidden_units = {}'.format(self.hidden_units))
            print(' - Activation = {}\n'.format(self.activation))
        bijectors=[]
        if(self.permute==True):
            _permutations = [np.random.permutation(self.n_params) for _ in range(self.n_maf)]
        else:
            _permutations=False

        for i in range(self.n_maf):
            # the permutation part comes from the code M. Raveri wrote,
            if _permutations:
                bijectors.append(tfb.Permute(_permutations[i].astype(np.int32)))
            # rest by myself
            bijectors.append(\
                tfb.MaskedAutoregressiveFlow(\
                shift_and_log_scale_fn=tfb.AutoregressiveNetwork(\
                params=2, event_shape=(self.n_params,), \
                hidden_units=self.hidden_units, \
                activation=self.activation, \
                kernel_initializer='glorot_uniform')))

        self.bijectors = bijectors