########################################################################
# Particle Metropolis-Hastings using gradient and Hessian information
# Copyright (c) 2014 Johan Dahlin ( johan.dahlin (at) liu.se )
#
# helpers.py
# Different helper files
#
########################################################################

import numpy as np
import os

#############################################################################################################################
# Calculate the Integrated Autocorrlation Time (disabled)
#############################################################################################################################   
def IACT(x):
    return 1.0;

#############################################################################################################################
# Calculate the Squared Jump distance
#############################################################################################################################   
def SJD(x):
    tmp = np.diff( x ) ** 2;
    out = np.sum( tmp );

    return out / ( len(x) - 1.0 );

#############################################################################################################################
# Calculate the log-pdf of a univariate Gaussian
#############################################################################################################################   
def loguninormpdf(x,mu,sigma):
    return -0.5 * np.log( 2.0 * np.pi * sigma**2) - 0.5 * (x-mu)**2 * sigma**(-2);

#############################################################################################################################
# Calculate the log-pdf of a multivariate Gaussian with mean vector mu and covariance matrix S
#############################################################################################################################   
def lognormpdf(x,mu,S):
    nx = len(S)
    norm_coeff = nx * np.log( 2.0 * np.pi ) + np.linalg.slogdet(S)[1]
    err = x-mu

    numerator = np.dot( np.dot(err,np.linalg.pinv(S)),err.transpose())
    return -0.5*(norm_coeff+numerator)

#############################################################################################################################
# Check if a matrix is positive semi-definite but checking for negative eigenvalues
#############################################################################################################################   
def isPSD(x):
    return np.all(np.linalg.eigvals(x) > 0)

#############################################################################################################################
# Print verbose progress reports from sampler  [TODO] Debug this!
#############################################################################################################################   
def verboseProgressPrint(kk,par,thp,th,aprob,accept,step,scoreP,score,infom,infomP,v):
    print("Reminder verboseProgressPrint: not debugged yet...")
    
    print("===========================================================================================")
    print("Iteration: " + str(kk) + " of " + str(par.nMCMC) + " complete.");
    print("===========================================================================================")
    print("Proposed parameters: " + str(thp) + " and current parameters: " + str(th) + ".");
    print("Acceptance probability: " + str(aprob) + " with outcome: " + str(accept) + ".");

    if (v==1) :
        print("Scaled score vector for proposed: " + str(step**2*0.5*scoreP) + " and curret: " + str(step**2*0.5*score) );

    if (v==2):
        print("Scaled score vector for proposed: " + str(step**2*0.5*scoreP/infomP) + " and current: " + str(step**2*0.5*score/infom) );
        print("Step size squared for proposed: " + str(step**2/infomP) + " and current: " + str(step**2/infom) );

    print("");


#############################################################################################################################
# Check if dirs for outputs exists, otherwise create them
#############################################################################################################################   
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    
#############################################################################################################################
# End of file
#############################################################################################################################  
