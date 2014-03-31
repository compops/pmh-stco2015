pmh2
====

Particle Metropolis-Hastings using gradient and Hessian information

This code was downloaded from < https://github.com/compops/pmh2 > or from < http://users.isy.liu.se/en/rt/johda87/ > and contains the code used to produce the results in the papers

* J. Dahlin, F. Lindsten and T. B. Schön, Particle Metropolis-Hastings using gradient and Hessian information. Pre-print arxiv:1311.0686v2 *

* J. Dahlin, F. Lindsten and T. B. Schön, Second-order particle MCMC for Bayesian parameter inference. Proceedings of the 18th World Congress of the International Federation of Automatic Control (IFAC), Cape Town, South Africa, August 2014. (accepted for publication) * 

* J. Dahlin, F. Lindsten and T. B. Schön, Particle Metropolis Hastings using Langevin Dynamics. Proceedings of the 38th International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Vancouver, Canada, May 2013. *

The papers are available as a preprint from < http://arxiv.org/abs/1311.0686v2 > and < http://users.isy.liu.se/en/rt/johda87/ >.

Requirements
--------------
The program is written in Python 2.7 and makes use of NumPy 1.7.1, SciPy 0.12.0, Matplotlib 1.2.1, Pandas. Please have these packages installed, on Ubuntu they can be installed using "sudo pip install --uppgrade **package-name** ".

Included files
--------------
**runmeLGSSfapf.py**
Generates a plot similar to Figure 3 in "Particle Metropolis-Hastings using gradient and Hessian information" using the fully-adapted particle filter and the fixed-lag smoother for estimating the negative Hessian. 

**runmeLGSSbpf.py**
Generates a plot similar to Figure 1 in "Second-order particle MCMC for Bayesian parameter inference" using the bootstrap particle filter and the filter smoother smoother for estimating the negative Hessian. 

**runmeHWSVbpf.py**
Generates a plot similiar to Figure 2 in "Second-order particle MCMC for Bayesian parameter inference" using the filter smoother for estimating the negative Hessian and the bootstrap particle filter.

Supporting files
--------------
**pmh.py**
Defines the general class for the particle MH algorithm and helper functions for this.

**smc.py**
Defines the general class for sequential Monte Carlo algorithm.

**classes.py**
Defines the different system models and generates the data.

**helpers.py**
Defines different helpers for the other functions.
