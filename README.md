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
## Normalizing Flow
Normalizing flows are implement in tensorflow as a chain of masked autoencoders and coordinate permutations. (More)

## Eigentension

## Parameter Difference in Update Form

## Goodness of Fit Degradation
