########################################################################
# Particle Metropolis-Hastings using gradient and Hessian information
# Copyright (c) 2014 Johan Dahlin ( johan.dahlin (at) liu.se )
#
# smc.py
# Sequential Monte Carlo samplers
#
########################################################################

import numpy as np
import scipy.weave as weave
from helpers import *
import matplotlib.pyplot as plt

class smcSampler(object):
 
    #############################################################################################################################
    # Initalisation
    #############################################################################################################################
    score = [];
    infom = [];   
    xhatf = [];
    xhatp = [];
    xhats = [];
    ll    = [];
    w     = [];
    a     = [];
    p     = [];
    v     = [];
    infom1 = [];
    infom2 = [];
    infom3 = [];

    #############################################################################################################################
    # Default settings
    #############################################################################################################################    
    nPart           = 100;
    Po              = 0;
    xo              = 0;
    so              = 0;
    fixedLag        = 5;
    resampFactor    = 1.0;
    onlydiagInfo    = 1;
    makeInfoPSD     = 0;    
    resamplingType  = "systematic"    
    filterType      = "bootstrap";
    smootherType    = "filtersmoother";

    #############################################################################################################################
    # Particle filtering: bootstrap particle filter
    #############################################################################################################################    
    def bPF(self,data,sys,par):
        a   = np.zeros((self.nPart,sys.T));    
        s   = np.zeros((self.nPart,sys.T));
        p   = np.zeros((self.nPart,sys.T));
        w   = np.zeros((self.nPart,sys.T));
        xh  = np.zeros((sys.T,1));
        sh  = np.zeros((sys.T,1));
        llp = 0.0;        
        
        p[:,0] = self.xo;        
        s[:,0] = self.so;       
        
        for tt in range(0, sys.T):
            if tt != 0:
                
                # Resample (if needed by ESS criteria)
                if ((np.sum(w[:,tt-1]**2))**(-1) < (self.nPart * self.resampFactor)):
                    
                    if self.resamplingType == "systematic":
                        nIdx = self.resampleSystematic(w[:,tt-1],par);
                    elif self.resamplingType == "multinomial":
                        nIdx = self.resampleMultinomial(w[:,tt-1],par);
                    else: 
                        nIdx = self.resample(w[:,tt-1],par);
                    
                    nIdx = np.transpose(nIdx.astype(int));
                else:
                    nIdx = np.arange(0,self.nPart);
                
                
                # Propagate
                s[:,tt] = sys.h(p[nIdx,tt-1], data.u[tt-1], s[nIdx,tt-1], tt-1)
                p[:,tt] = sys.f(p[nIdx,tt-1], data.u[tt-1], s[:,tt], data.y[tt-1], tt-1) + sys.fn(p[nIdx,tt-1], s[:,tt], data.y[tt-1], tt-1) * np.random.randn(1,self.nPart);
                a[:,tt] = nIdx;
            
            # Calculate weights
            w[:,tt] = loguninormpdf(data.y[tt], sys.g(p[:,tt], data.u[tt], s[:,tt], tt), sys.gn(p[:,tt], s[:,tt], tt) );
            wmax    = np.max(w[:,tt]);
            w[:,tt] = np.exp(w[:,tt] - wmax);
            
            # Estimate log-likelihood
            llp += wmax + np.log(np.sum(w[:,tt])) - np.log(self.nPart);
            
            # Estimate state
            w[:,tt] /= np.sum(w[:,tt]);
            xh[tt]  = np.sum( w[:,tt] * p[:,tt] );
            sh[tt]  = np.sum( w[:,tt] * s[:,tt] );
        
        self.xhatf = xh;
        self.shatf = sh;
        self.ll    = llp;
        self.w     = w;
        self.a     = a;
        self.p     = p;
        self.s     = s;

    #############################################################################################################################
    # Particle filtering: fully-adapted filter
    #############################################################################################################################    
    def faPF(self,data,sys,par):
        a   = np.zeros((self.nPart,sys.T));    
        s   = np.zeros((self.nPart,sys.T));        
        p   = np.zeros((self.nPart,sys.T));        
        w   = np.zeros((self.nPart,sys.T));
        v   = np.zeros((self.nPart,sys.T));
        xh  = np.zeros((sys.T,1));
        sh  = np.zeros((sys.T,1));
        llp = loguninormpdf( data.y[0], 0, sys.gn(0, 0, 0) );  
        
        p[:,0] = self.xo;        
        s[:,0] = self.so;       
        
        for tt in range(0, sys.T):
            if tt != 0:
                # Resample (if needed by ESS criteria)
                if ((np.sum(w[:,tt-1]**2))**(-1) < (self.nPart * self.resampFactor)):
                    
                    if self.resamplingType == "systematic":
                        nIdx = self.resampleSystematic(v[:,tt-1],par);
                    elif self.resamplingType == "multinomial":
                        nIdx = self.resampleMultinomial(v[:,tt-1],par);
                    else: 
                        nIdx = self.resample(v[:,tt-1],par);
                    
                    nIdx = np.transpose(nIdx.astype(int));
                else:
                    nIdx = np.arange(0,self.nPart);
                
                # Propagate
                s[:,tt] = sys.h(p[nIdx,tt-1], data.u[tt-1], s[nIdx,tt-1], tt-1)
                p[:,tt] = sys.fa(p[nIdx,tt-1], data.y[tt], data.u[tt-1], s[:,tt], tt-1) + sys.fna(p[nIdx,tt-1], data.y[tt], s[:,tt], tt-1) * np.random.randn(1,self.nPart);
                a[:,tt] = nIdx;
            
            if tt != (sys.T-1):
                # Calculate weights
                v[:,tt] = loguninormpdf(data.y[tt+1], sys.ga(p[:,tt], data.u[tt], s[:,tt], tt), sys.gna(p[:,tt], s[:,tt], tt) );
                vmax    = np.max(v[:,tt]);
                v[:,tt] = np.exp(v[:,tt] - vmax);
                
                # Estimate log-likelihood
                llp += vmax + np.log(np.sum(v[:,tt])) - np.log(self.nPart);
                
                # Normalise the weights                
                v[:,tt] /= np.sum( v[:,tt] );
            
            # Calculate the normalised filter weights (1/N) as it is a FAPF
            w[:,tt] = np.ones(self.nPart) / self.nPart;
            
            # Estimate the state
            xh[tt] = np.sum( w[:,tt] * p[:,tt] );
            sh[tt] = np.sum( w[:,tt] * s[:,tt] );
            
        self.xhatf = xh;
        self.shatf = sh;        
        self.ll    = llp;
        self.w     = w;
        self.v     = v;
        self.a     = a;
        self.p     = p;
        self.s     = s;
    
    #############################################################################################################################
    # Particle smoothing: fixed-lag smoother
    #############################################################################################################################
    def flPS(self,data,sys,par):
        
        # Put all the recursions to start at zero (for GARCH)
        sys.rD1  = np.zeros((self.nPart,par.nPars));
        sys.rDD1 = np.zeros((self.nPart,par.nPars));
        sys.rDD2 = np.zeros((self.nPart,par.nPars));
        
        # Run the initial filter
        if (self.filterType == "bootstrap"):
            self.bPF(data,sys,par);
        elif ( (self.filterType == "fullyadapted") & (sys.supportsFA==1.0) ):
            self.faPF(data,sys,par);
        else:
            raise NameError("Unknown or incompatable filter selected.");
         
        ## Run smoother
        xs    = np.zeros((sys.T,1));
        ss    = np.zeros((sys.T,1));
        sa    = np.zeros((self.nPart,par.nPars));
        s1    = np.zeros((par.nPars));
        s2    = np.zeros((par.nPars,par.nPars));
        info  = np.zeros((par.nPars,par.nPars));
        info2 = np.zeros((par.nPars,par.nPars));
        
        #---------------------------------------------------------------------------------------------------------
        # Version 1: Use the filter-smoother for the element in the Louis identity that we can't compute
        #---------------------------------------------------------------------------------------------------------
        
        if ( self.flVersion == "filtersmoother" ):
            for tt in range(0, sys.T-1):
                at  = np.arange(0,self.nPart)
                kk  = np.min( (tt+self.fixedLag, sys.T-1) )
                
                # Reconstruct particle trajectory
                for ii in range(kk,tt,-1):
                    att = at.astype(int);
                    at  = at.astype(int);
                    at  = self.a[at,ii];
                    at  = at.astype(int);
                
                # Get the ancestor indicies for the filter smoother
                bt = self.a[:,tt+1]; bt = bt.astype(int);
            
                # Estimate state
                xs[tt] = np.sum( self.p[at,tt] * self.w[:, kk] );
                ss[tt] = np.sum( self.s[at,tt] * self.w[:, kk] );
                
                # Estimate score
                tmp1 = sys.Dparm(  self.p[att,tt+1], self.p[at,tt], data.y[tt], self.s[at,tt], at, par);
                tmp2 = sys.DDparm( self.p[att,tt+1], self.p[at,tt], data.y[tt], self.s[at,tt], at, par);
                
                osa  = sa;
                sa   = osa[bt,:] + sys.Dparm(  self.p[:,tt+1], self.p[bt,tt], data.y[tt], self.s[at,tt], at, par);
                
                for nn in range(0,par.nPars):
                    s1[nn]     += np.sum( self.w[ : ,kk ] * tmp1[:,nn] );
                    for mm in range(0,par.nPars):
                        s2[nn,mm]  += np.sum( self.w[ : ,kk ] * tmp2[:,nn,mm] );
                    
                
            # Calculate the second term in the Louis identity and the score
            for nn in range(0,self.nPart):
                info2 += self.w[nn, sys.T-1] * ( np.mat(sa[nn,:]).T * np.mat(sa[nn,:]) );            
            
            # Calculate the information matrix
            info = np.mat(s1).T * np.mat(s1) - ( s2 + info2 );
        
        #---------------------------------------------------------------------------------------------------------
        # Version 2: Neglect the cross-terms in the Loius identity
        #---------------------------------------------------------------------------------------------------------
        
        if ( self.flVersion == "neglectcross" ):
            for tt in range(0, sys.T-1):
                at  = np.arange(0,self.nPart)
                kk  = np.min( (tt+self.fixedLag, sys.T-1) )
                
                # Reconstruct particle trajectory
                for ii in range(kk,tt,-1):
                    att = at.astype(int);
                    at  = at.astype(int);
                    at  = self.a[at,ii];
                    at  = at.astype(int);
                
                # Estimate state
                xs[tt] = np.sum( self.p[at,tt] * self.w[:, kk] );
                ss[tt] = np.sum( self.s[at,tt] * self.w[:, kk] );
                
                # Estimate score
                tmp1 = sys.Dparm( self.p[att,tt+1], self.p[at,tt], data.y[tt], self.s[at,tt], at, par);
                tmp2 = sys.DDparm( self.p[att,tt+1], self.p[at,tt], data.y[tt], self.s[at,tt], at, par)
                
                for nn in range(0,par.nPars):            
                    s1[nn]       += np.sum( tmp1[:,nn] * self.w[:,kk] );
                    for mm in range(0,par.nPars):  
                        s2[nn,mm]    += np.sum( tmp2[:,nn,mm] * self.w[:,kk] );
                        info2[nn,mm] += np.sum( tmp1[:,nn] * tmp1[:,mm] * self.w[:,kk] );
                
            # Calculate the information matrix
            info = np.mat(s1).T * np.mat(s1) - ( s2 + info2 );
        
        #---------------------------------------------------------------------------------------------------------
        # Version 3: Use the fixed-lag smooother for everything
        #---------------------------------------------------------------------------------------------------------
        
        if ( self.flVersion == "full" ):
            
            sal = np.zeros((self.nPart,par.nPars,sys.T))
            
            # construct the sa from the filtersmoother
            for tt in range(0, sys.T-1):
                # Get the ancestor indicies for the filter smoother
                bt = self.a[:,tt+1]; bt = bt.astype(int);
                
                # Estimate the alpha quantity
                sal[:,:,tt+1] = sal[bt,:,tt] + sys.Dparm( self.p[:,tt+1], self.p[bt,tt], data.y[tt], self.s[bt,tt], bt, par );
            
            # Put all the recursions to start at zero (for GARCH)
            sys.rD1  = np.zeros((self.nPart,par.nPars));
            sys.rDD1 = np.zeros((self.nPart,par.nPars));
            sys.rDD2 = np.zeros((self.nPart,par.nPars));
            
            # Run the fixed-lag smoother for the rest
            for tt in range(0, sys.T-1):
                at  = np.arange(0,self.nPart)
                kk  = np.min( (tt+self.fixedLag, sys.T-1) )
                
                # Reconstruct particle trajectory
                for ii in range(kk,tt,-1):
                    att = at.astype(int);
                    at  = at.astype(int);
                    at  = self.a[at,ii];
                    at  = at.astype(int);
                
                # Estimate state
                xs[tt] = np.sum( self.p[at,tt] * self.w[:, kk] );
                ss[tt] = np.sum( self.s[at,tt] * self.w[:, kk] );                
                
                # Estimate score
                tmp1 = sys.Dparm( self.p[att,tt+1], self.p[at,tt], data.y[tt], self.s[at,tt], at, par);
                tmp2 = sys.DDparm( self.p[att,tt+1], self.p[at,tt], data.y[tt], self.s[at,tt], at, par)
                
                for nn in range(0,par.nPars):            
                    s1[nn]       += np.sum( tmp1[:,nn] * self.w[:,kk] );

                    for mm in range(0,(nn+1)):  
                        s2[nn,mm]    += np.sum( self.w[:,kk] * tmp2[:,nn,mm] );
                        info2[nn,mm] += np.sum( self.w[:,kk] * ( tmp1[:,nn]*tmp1[:,mm] + tmp1[:,nn]*sal[at,mm,tt] + sal[at,nn,tt]*tmp1[:,mm] ) );
                        #info2[nn,mm] += np.sum( self.w[:,kk] * tmp1[:,nn]    * tmp1[:,mm]    );
                        #info2[nn,mm] += np.sum( self.w[:,kk] * tmp1[:,nn]    * sal[at,mm,tt] );
                        #info2[nn,mm] += np.sum( self.w[:,kk] * sal[at,nn,tt] * tmp1[:,mm]    );
                        
                        s2[mm,nn]     = s2[nn,mm];
                        info2[mm,nn]  = info2[nn,mm];
            
            # Calculate the information matrix
            info = np.mat(s1).T * np.mat(s1) - ( s2 + info2 );

        #---------------------------------------------------------------------------------------------------------
        # Write output
        #---------------------------------------------------------------------------------------------------------
        
        # Remove score if we would lite to run without it
        if ( par.zeroscore == 1):
            s1 = 0; 
        
        self.xhats  = xs;
        self.shats  = ss;
        self.score  = s1;
        self.infom  = info;
        self.infom1 = np.mat(s1).T * np.mat(s1);
        self.infom2 = s2;
        self.infom3 = info2;
       
    #############################################################################################################################
    # Particle smoother: filter smooothing
    #############################################################################################################################       
    def fsPS(self,data,sys,par):
        
        # Put all the recursions to start at zero (for GARCH)
        sys.rD1  = np.zeros((self.nPart,par.nPars));
        sys.rDD1 = np.zeros((self.nPart,par.nPars));
        sys.rDD2 = np.zeros((self.nPart,par.nPars));
        
        # Run the initial filter
        if (self.filterType == "bootstrap"):
            self.bPF(data,sys,par);
        elif ( (self.filterType == "fullyadapted") & (sys.supportsFA==1.0) ):
            self.faPF(data,sys,par);
        else:
            raise NameError("Unknown or incompatable filter selected.");
        
        ss    = np.zeros((sys.T,1));
        xs    = np.zeros((sys.T,1));
        sa    = np.zeros((self.nPart,par.nPars));
        sb    = np.zeros((self.nPart,par.nPars,par.nPars));
        score = np.zeros((par.nPars));
        info  = np.zeros((par.nPars,par.nPars));
        info2 = np.zeros((par.nPars,par.nPars));
        info3 = np.zeros((par.nPars,par.nPars));
        
        for tt in range(0, sys.T-1):
            # Get the ancestor indicies
            at = self.a[:,tt+1]; at = at.astype(int);
            
            # Estimate state
            xs[tt] = np.sum( self.p[:, tt] * self.w[:, sys.T-1] );
            ss[tt] = np.sum( self.s[:, tt] * self.w[:, sys.T-1] );
            
            # Save the old quantaties
            osa = sa; osb = sb;
            
            # Estimate score and take the resampling into accoung
            sa  = osa[at,:] + sys.Dparm(  self.p[:,tt+1], self.p[at,tt], data.y[tt], self.s[at,tt], at, par);
            sb  = osb[at,:] + sys.DDparm( self.p[:,tt+1], self.p[at,tt], data.y[tt], self.s[at,tt], at, par);
        
        # Calculate the second term in the Louis identity and the score
        for nn in range(0,self.nPart):
            info2 += self.w[nn, sys.T-1] * np.mat(sa[nn,:]).T * np.mat(sa[nn,:]);
            info3 += self.w[nn, sys.T-1] * sb[nn,:,:];
            score += self.w[nn, sys.T-1] * sa[nn,:];
        
        # Calculate information matrix
        info  = np.mat(score).T * np.mat(score) - info2 - info3;

        if ( par.zeroscore == 1):
            score = 0; 
        
        self.shats = ss;
        self.xhats = xs;
        self.score = score;
        self.infom = info;
        self.infom1 = np.mat(score).T * np.mat(score);
        self.infom2 = info2;
        self.infom3 = info3;
    
    def plotTrajectories(self,sys):
        plt.plot(sys.T-1*np.ones(self.nPart),self.p[:,sys.T-1],'k.'); plt.axis((0,sys.T,-2,2))
        
        plt.hold("on");
        
        # Plot all the particles and their resampled ancestors
        for ii in range(0,self.nPart):
            att = ii;
            for tt in np.arange(sys.T-2,0,-1):
                at = self.a[att,tt+1]; at = at.astype(int);
                plt.plot(tt,self.p[at,tt],'k.');
                plt.plot((tt,tt+1),(self.p[at,tt],self.p[att,tt+1]),'k');
                att = at; att = att.astype(int);
        
        plt.hold("off")
    
    #############################################################################################################################
    # Resampling helpers
    #############################################################################################################################
    def resample(self,w,par):
        code = \
        """ int j = 0;
            double csw = 0;
            for(int k = 0; k < H; k++)
            {
                while(csw < u && j < H - 1)
                {
                    j++;
                    csw += w(j);
                }
                Ind(k) = j;
                u = u + 1.;
            }
        """ 
        u = float(np.random.uniform())
        H = self.nPart;
        w = np.double(w * H / sum(w))
        Ind = np.empty(H, dtype='int')
        weave.inline(code,['u','H','w','Ind'], type_converters=weave.converters.blitz)
        return Ind;
    
    def resampleSystematic(self,w,par):
        code = \
        """ int j = 0;
            for(int k = 0; k < H; k++)
            {
                double uu  = ( u + k ) / H;
                
                while( w(j) < uu && j < H - 1)
                {
                    j++;
                }
                Ind(k) = j;
            }
        """ 
        H = self.nPart;
        u = float( np.random.uniform() );
        w = np.cumsum(w) / sum(w);
        Ind = np.empty(H, dtype='int')
        weave.inline(code,['u','H','w','Ind'], type_converters=weave.converters.blitz)
        return Ind;
    
    def resampleMultinomial(self,w,par):
        code = \
        """ for(int k = 0; k < H; k++)  // For each particle
            {
                int j = 0;
                
                while( w(j) < u(k) && j < H - 1)
                {
                    j++;
                }
                Ind(k) = j;
            }
        """ 
        H = self.nPart;       
        u = np.random.uniform(0.0,1.0,H);
        
        w = np.cumsum(w) / sum(w);
        Ind = np.empty(H, dtype='int')
        weave.inline(code,['u','H','w','Ind'], type_converters=weave.converters.blitz)
        return Ind;
    
#############################################################################################################################
# End of file
#############################################################################################################################        
