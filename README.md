# Tension Calibration
Code to calibrate cosmological tensions.

To Do:
- [ ] Readme
- [x] Normalizing flow
- [x] Eigentension
- [x] Parameter difference in update form
- [ ] Goodness of fit degredation (unexpected optimizer results)

To complete the calibration you will need to provide chains (on the order of 1-5k). I have used the following emulators to create these chains:
- Cosmopower for CMB emulation. I have a version that uses Emcee instead of the GPU affine sampler used in the example.
- LSST emulator made by Supranta.
- (more in the future!)

The affine sampler, although fast, seems to leave many gaps in the parameter space. The GPU runs out of memory for chains longer than about 3000 samples. All in all, we cannot run chains long enough to let the walkers fully explore the parameter space.

In the notebooks folder you can see examples of how to compute each tension metric as well as a proof-of-concept example for tension error analysis. The code is meant to simplify notebooks that require these to be computed repeatedly and to standardize the implementation of each metric for each data set.

---

# Implementation
Within "notebooks/metrics.ipynb" you can find definitions of each of these metrics.
## Normalizing Flow
Normalizing flows are implement in tensorflow as a chain of masked autoencoders and coordinate permutations. Since we are integrating over the entire parameter space the order of the parameters should not matter, but allowing the parameters to permute can allow the NN to learn more expressive densities. The NN is made with 2 hidden layers of $2d$ units (d is the dimension of the parameter space). The number of bijectors in the chain is also $2d$. These can be tuned as needed.

## Eigentension
This is an interesting one. First, diagonalize the covariance matrix of one experiment and change to that basis. Determine if an eigenvector is "well measured" by taking the ratio of the variance in the posterior to the variance in the prior. Keep only the well-measured eigenvectors and compute any tension metric.

## Parameter Difference in Update Form
To do this we need joint chains for two experiments. The joint chains are sampled using MCMC with log-probability as the sum of log-priors and log-likelihoods.

## Goodness of Fit Degradation
We use the pybobyqa optimizer to find the parameter vector $\theta_{max}$ at the global maximum of the posterior, then compute the likelihood at $\theta_{max} using the corresponding emulator.
