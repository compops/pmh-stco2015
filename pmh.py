########################################################################
# Particle Metropolis-Hastings using gradient and Hessian information
# Copyright (c) 2014 Johan Dahlin ( johan.dahlin (at) liu.se )
#
# pmh.py
# Particle Metropolis-Hastings samplers
#
########################################################################

import numpy as np
from helpers import *
import pandas

class stPMH(object):
               
    #############################################################################################################################
    # Run main sampler
    #############################################################################################################################           
    
    def runSampler(self,sm,data,sys,thSys,par,PMHtype):

        #############################################################################################################################
        # Initalisation
        #############################################################################################################################
        
        self.iter    = 0;
        self.PMHtype = PMHtype;
        self.nPars   = par.nPars;

        # Allocate vectors
        self.ll             = np.zeros((par.nMCMC,1))
        self.llp            = np.zeros((par.nMCMC,1))
        self.th             = np.zeros((par.nMCMC,par.nPars))
        self.tho            = np.zeros((par.nMCMC,par.nPars))
        self.thp            = np.zeros((par.nMCMC,par.nPars))
        self.aprob          = np.zeros((par.nMCMC,1))
        self.accept         = np.zeros((par.nMCMC,1))
        self.score          = np.zeros((par.nMCMC,par.nPars))
        self.scorep         = np.zeros((par.nMCMC,par.nPars))
        self.infom          = np.zeros((par.nMCMC,par.nPars,par.nPars))
        self.infomp         = np.zeros((par.nMCMC,par.nPars,par.nPars))
        self.prior          = np.zeros((par.nMCMC,1))
        self.priorp         = np.zeros((par.nMCMC,1))
        self.J              = np.zeros((par.nMCMC,1))        
        self.Jp             = np.zeros((par.nMCMC,1))        
        self.proposalProb   = np.zeros((par.nMCMC,1))        
        self.proposalProbP  = np.zeros((par.nMCMC,1))        
        self.llDiff         = np.zeros((par.nMCMC,1))        
        
        withinNormLimit = np.zeros((par.nMCMC,1))
        
        # Get the order of the PMH sampler
        if ( PMHtype == "PMH0" ):
            self.PMHtypeN = 0;
        elif ( PMHtype == "PMH1" ):
            self.PMHtypeN = 1;
        elif ( PMHtype == "PMH2" ):
            self.PMHtypeN = 2;
        
        # Initialise the parameters in the proposal
        thSys.storeParameters(par.initPar,sys,par);
    
        # Run the initial filter/smoother
        self.evaluateSMC(sm,data,thSys,par);
        self.acceptParameters(thSys,par);
    
        # Inverse transform and then save the initial parameters and the prior
        self.tho[0,:]  = thSys.returnParameters(par);
        
        thSys.invTransform();
        self.J[0]     = thSys.Jacobian();
        self.prior[0] = thSys.prior();
        self.th[0,:]  = thSys.returnParameters(par);
    
        #############################################################################################################################
        # Main MCMC-loop
        #############################################################################################################################
        for kk in range(1,par.nMCMC):
    
            self.iter = kk;
            
            # Propose parameters
            self.sampleProposal(par);
            thSys.storeParameters(self.thp[kk,:],sys,par);
            thSys.transform();
    
            # Calculate acceptance probability
            self.calculateAcceptanceProbability(sm, data, thSys, par);
            
            # Accept/reject step
            if (np.random.random(1) < self.aprob[kk]):
                self.acceptParameters(thSys,par);             
            else:
                self.rejectParameters(thSys,par);
            
            # Write out progress report
            if np.remainder(kk,par.nProgressReport) == 0:
                self.progressPrint();
    
            # Check the distance to the true parameters
            if ( ( np.linalg.norm( self.tho[kk,:] - sys.par, 2) < par.normLimit ) & ( self.accept[kk] == 1.0 ) ):
                withinNormLimit[kk] = 1.0;

        #############################################################################################################################
        # Compile the output information
        #############################################################################################################################    
        
        # Find the iteration where the first crossing of the ball with radius par.normLimit occurs
        try:
            firstCross = next(idx for idx, value in enumerate(withinNormLimit) if value == 1.0)
            firstCross = np.min( (firstCross, par.nMCMC) );
        except:
            firstCross = np.nan;
            
        # Save the data for later to class
        self.firstCross = firstCross;
            
    #############################################################################################################################
    # Compile the results
    #############################################################################################################################
    def writeToFile(self,filename,par):
        # Calculate the natural gradient
        ngrad = np.zeros((par.nMCMC,par.nPars));
        
        if ( self.PMHtype == "PMH1" ):
            ngrad = self.score;
        elif ( self.PMHtype == "PMH2" ):
            for kk in range(0,par.nMCMC):
                ngrad[kk,:] = np.dot( self.score[kk,:], np.linalg.pinv(self.infom[kk,:,:]) );

        # Construct the columns labels
        columnlabels = [None]*(3*par.nPars+3);
        for ii in xrange(3*par.nPars+3):  columnlabels[ii] = ii;

        for ii in range(0,par.nPars):
            columnlabels[ii]              = "th" + str(ii); 
            columnlabels[ii+par.nPars]    = "thp" + str(ii); 
            columnlabels[ii+2*par.nPars]  = "ng" + str(ii); 
        
        # Compile the results for output
        out = np.hstack((self.th,self.thp,ngrad,self.aprob,self.ll,self.accept));
    
        # Write out the results to file
        tmp = pandas.DataFrame(out,columns=columnlabels);
        if (filename == ""):
            filename = 'resultsRaw/' + str(par.fileprefix) + '/PMH' + str(self.PMHtypeN) + str(sm.filterType) +  '_' + str(sm.smootherType) + '/' + str(par.dataset) + '_' + str(sm.nPart) + '.csv';
        
        ensure_dir(filename);
        tmp.to_csv(filename);
 
    #############################################################################################################################
    # Sample the proposal
    #############################################################################################################################
    def sampleProposal(self,par):
        
        step = par.stepSize[ self.PMHtypeN ];
        
        if ( self.PMHtype == "PMH0" ):
            self.thp[self.iter] = self.th[self.iter-1,:] + np.random.multivariate_normal( np.zeros(par.nPars), step**2 * np.eye( par.nPars )   );
        elif ( self.PMHtype == "PMH1" ):
            self.thp[self.iter] = self.th[self.iter-1,:] + 0.5 * step**2 *        self.score[self.iter-1,:]                         + np.random.multivariate_normal(np.zeros(par.nPars), step**2 * np.eye( par.nPars )   );
        elif ( self.PMHtype == "PMH2" ):
            self.thp[self.iter] = self.th[self.iter-1,:] + 0.5 * step**2 * np.dot(self.score[self.iter-1,:], np.linalg.pinv(self.infom[self.iter-1,:,:])) + np.random.multivariate_normal(np.zeros(par.nPars), step**2 * np.linalg.pinv(self.infom[self.iter-1,:,:]) );

    #############################################################################################################################
    # Calculate Acceptance Probability
    #############################################################################################################################
    def calculateAcceptanceProbability(self, sm, data, thSys, par):
    
        step = par.stepSize[ self.PMHtypeN ];
        
        # Check the "hard prior"
        if (thSys.priorUniform() == 0.0):
            if (par.writeOutPriorWarnings):
                print("The parameters " + str(thp[kk,:]) + " were proposed.");
            return np.zeros(3);

        # Run the smoother to get the ll-estimate, score and infom-estimate
        self.evaluateSMC(sm,data,thSys,par);

        # Compute the part in the acceptance probability related to the non-symmetric proposal
        if ( self.PMHtype == "PMH0" ):
            proposalP = 0;
            proposal0 = 0;
        elif ( self.PMHtype == "PMH1" ):
            proposalP = lognormpdf( self.thp[self.iter,:], self.th[self.iter-1,:]  + 0.5 * step**2 * self.score[self.iter-1,:],                                  step**2 * np.eye( par.nPars )    );
            proposal0 = lognormpdf( self.th[self.iter-1,:],  self.thp[self.iter,:] + 0.5 * step**2 * self.scorep[self.iter,:],                                   step**2 * np.eye( par.nPars )    );
        elif ( self.PMHtype == "PMH2" ):
            proposalP = lognormpdf( self.thp[self.iter,:], self.th[self.iter-1,:]  + 0.5 * step**2 * np.dot( self.score[self.iter-1,:],  np.linalg.pinv(self.infom[self.iter-1,:,:])  ), step**2 * np.linalg.pinv(self.infom[self.iter-1,:,:])  );
            proposal0 = lognormpdf( self.th[self.iter-1,:],  self.thp[self.iter,:] + 0.5 * step**2 * np.dot( self.scorep[self.iter,:],   np.linalg.pinv(self.infomp[self.iter,:,:]) ), step**2 * np.linalg.pinv(self.infomp[self.iter,:,:]) );
        
        # Compute prior and Jacobian
        self.priorp[ self.iter ]    = thSys.prior();
        self.Jp[ self.iter ]        = thSys.Jacobian();

        # Compute the acceptance probability
        self.aprob[ self.iter ] = self.flag * np.exp( self.llp[ self.iter, :] - self.ll[ self.iter-1, :] + proposal0 - proposalP + self.priorp[ self.iter, :] - self.prior[ self.iter-1, :] + self.Jp[ self.iter, :] - self.J[ self.iter-1, :] );
        
        # Store the proposal calculations
        self.proposalProb[ self.iter ]  = proposal0;
        self.proposalProbP[ self.iter ] = proposalP;
        self.llDiff[ self.iter ]        = self.llp[ self.iter, :] - self.ll[ self.iter-1, :];
        
    #############################################################################################################################
    # Calculate LL for PMH0-function
    #############################################################################################################################   
    def evaluateSMC(self,sm,data,thSys,par):
        
        #########################################################################################################################
        # PMH0
        #########################################################################################################################
        # If PMH0, only run the filter and return the log-likelihood
        if ( self.PMHtype == "PMH0"):
            if ( sm.filterType == "kalman" ):
                sm.filter(data,thSys,par);
            elif ( sm.filterType == "bootstrap" ):
                sm.bPF(data,thSys,par);
            elif ( ( sm.filterType == "fullyadapted" ) & (thSys.supportsFA==1.0) ):
                sm.faPF(data,thSys,par);
            else:
                raise NameError("No appropriate filter selected. Check if the model can be fully adapted or if the Kalman methods can be used");
            
            self.llp[ self.iter ]        = sm.ll;
            self.scorep[ self.iter,: ]   = np.zeros(par.nPars);
            self.infomp[ self.iter,:,: ] = np.zeros((par.nPars,par.nPars));
            self.flag                    = 1.0
            return None;

        #########################################################################################################################
        # PMH1 or 2
        #########################################################################################################################
        
        # Run the correct smoother
        if ( sm.smootherType == "kalman" ):
            print("KS: Not supported...");
        elif ( sm.smootherType == "fixedlag" ):
            sm.flPS(data,thSys,par);
        elif ( sm.smootherType == "ffbsm" ):
            sm.ffbsmPS(data,thSys,par);
        elif ( sm.smootherType == "filtersmoother"):
            sm.fsPS(data,thSys,par)
        else:
            raise NameError("No appropriate smoother selected. Check if the model can be fully adapted or if the Kalman methods can be used");
    
        #########################################################################################################################
        # PMH2: Check tthat the information matrix is Postive semi-definite
        #########################################################################################################################        
        
        # Remove off-diagonal elements if needed
        if (sm.onlydiagInfo):
            infom = np.diag( np.diag( sm.infom ) );
        else:
            infom = sm.infom;
        
        # Check for NaN-elements or Inf-elements
        if ( ( np.sum( np.isnan(infom) ) == 0 ) | ( np.sum( np.isinf(infom) ) == 0 ) ):
            
            # Check if the infomatris is PSD
            if ( ( self.PMHtype == "PMH2" ) & ( ~isPSD( infom ) ) ):
                
                if (sm.makeInfoPSD):
                    # Add a diagonal matrix proportional to the largest negative eigenvalue to make it PSD
                    #print("I have a problem with a non-PSD-matrix, mirroring eigenvalues at: " + str( thSys.par[0] ) + " and th1: " + str( thSys.par[1] ) );
                    mineigenvalue = np.min( np.linalg.eig(sm.infom)[0] )
                    infom = sm.infom - 2 * mineigenvalue * np.eye( self.nPars )
                    flag = 1.0;
                    
                    if ~isPSD( infom ):
                        print("Added diagonal matrix to get PSD, but it did not work...");
                    
                else:
                    # Do nothing and discard the information matrix
                    print("I have a problem with a non-PSD-matrix, discarding information matrix: " + str( thSys.par[0] ) + " and th1: " + str( thSys.par[1] ) );
                    flag = 0.0;
                
            else:
                # The information matrix is PSD or we have PMH1 and do not care
                flag = 1.0;
            
        else:
            # Change the new likelihood to -inf to reject the parameter
            sm.ll = - np.inf;
            print("I have problem with NaN-elements or Inf-elements in the information matrix, rejecting proposed parameter at iteration: " + str(self.iter) + ".")
            print(infom)
        
        
        #########################################################################################################################
        # Generate output
        #########################################################################################################################    
        self.llp[ self.iter ]        = sm.ll;
        self.scorep[ self.iter,: ]   = sm.score;
        self.infomp[ self.iter,:,: ] = infom;
        self.flag                    = flag

    def acceptParameters(self,thSys,par):        
        self.th[self.iter,:]      = self.thp[self.iter,:];
        self.tho[self.iter,:]     = thSys.returnParameters(par);
        self.ll[self.iter]        = self.llp[self.iter];
        self.score[self.iter,:]   = self.scorep[self.iter,:];
        self.infom[self.iter,:,:] = self.infomp[self.iter,:];
        self.accept[self.iter]    = 1.0;
        self.prior[self.iter,:]   = self.priorp[self.iter,:];
        self.J[self.iter,:]       = self.Jp[self.iter,:];   
        
    def rejectParameters(self,thSys,par):
        self.th[self.iter,:]      = self.th[self.iter-1,:];
        self.tho[self.iter,:]     = self.tho[self.iter-1,:];
        self.ll[self.iter]        = self.ll[self.iter-1];
        self.prior[self.iter,:]   = self.prior[self.iter-1,:]
        self.score[self.iter,:]   = self.score[self.iter-1,:];
        self.infom[self.iter,:,:] = self.infom[self.iter-1,:,:];
        self.J[self.iter,:]       = self.J[self.iter-1,:];   
    
    #############################################################################################################################
    # Print small progress reports
    #############################################################################################################################   
    def progressPrint(self):
        print( self.iter, np.mean(self.accept[0:self.iter]), self.th[self.iter,:], np.mean(self.th[0:self.iter,:], axis=0) );
    
    def calculateESS(self,par):
        ESSout = np.zeros(par.nPars);
        
        for ii in range(0,par.nPars):
            ESSout[ii] = ( par.nMCMC - par.nBurnIn ) / IACT( self.th[par.nBurnIn:par.nMCMC,ii] );
        
        return(ESSout);
    
#############################################################################################################################
# End of file
#############################################################################################################################
