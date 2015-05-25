from smc import *
from pmh import *
from classes import *
from helpers import *
import pandas
import numpy as np
import matplotlib.pyplot as plt

########################################################################
# Arrange the data structures
########################################################################
data             = stData();
smc              = smcSampler();
par              = stParameters();
pmh0             = stPMH();
pmh1             = stPMH();
pmh2             = stPMH();

########################################################################
# Setup the system
########################################################################
sys              = stSystemHW()
sys.version      = "standard"
sys.par          = np.zeros((3,1))
sys.par[0]       = 0.98;
sys.par[1]       = 0.16;
sys.par[2]       = 0.65;
sys.T            = 250;

thSys            = stSystemHW()
thSys.T          = sys.T;
thSys.version    = sys.version

########################################################################
# Setup the parameters for the algorithm
########################################################################

par.fileprefix   = "hw"

par.nPars          = 2;
par.nMCMC          = 150;
par.nBurnIn        = 0;
par.normLimit      = 0.1;
par.verboseSampler = 0;

smc.nPart          = 5000;
smc.resamplingType = "systematic";     # multinomial or systematic
smc.filterType     = "bootstrap";      # bootstrap or fullyadapted
smc.smootherType   = "fixedlag";       # filtersmoother or fixedlag 
smc.flVersion      = "filtersmoother"; # filtersmoother or full
smc.fixedLag       = 12;
smc.onlydiagInfo   = 1;
smc.makeInfoPSD    = 1;

########################################################################
# Select the initial point for the sampler
########################################################################

par.initPar      = np.zeros((3,1))
par.initPar[0]   = 0.1;
par.initPar[1]   = 2.0;
par.initPar[2]   = 0.1;

########################################################################
# Generate or read the data
########################################################################

par.dataset = 0;
file = 'data/' + str(par.fileprefix) + 'T' + str(sys.T) +'/' + str(par.fileprefix) + 'DataT' + str(sys.T) + str(par.dataset) + '.csv'
tmp = np.loadtxt(file,delimiter=",")
data.x = tmp[:,0]; data.u = tmp[:,1]; data.y = tmp[:,2];

########################################################################
# Run the samplers
########################################################################

# Set the step sizes
par.stepSize = (0.05, 0.045, 1.70);

pmh0.runSampler(smc, data, sys, thSys, par, "PMH0");
pmh1.runSampler(smc, data, sys, thSys, par, "PMH1");
pmh2.runSampler(smc, data, sys, thSys, par, "PMH2");

# Uncomment to export data to file
#pmh0.writeToFile("results/hwsv-bpf/pmh0.csv",par);
#pmh1.writeToFile("results/hwsv-bpf/pmh1.csv",par);
#pmh2.writeToFile("results/hwsv-bpf/pmh2.csv",par);

########################################################################
# Plot the results
########################################################################

plt.subplot(3,1,1); 
plt.plot(pmh0.th[:,0],pmh0.th[:,1],'k'); xlabel("th0"); ylabel("th1"); 
plt.hold("on"); 
plt.plot(pmh0.th[:,0],pmh0.th[:,1],'k.');
plt.plot([sys.par[0],sys.par[0]],[0,2],'k:')
plt.plot([0,1],[sys.par[1],sys.par[1]],'k:')
plt.hold("off"); 
plt.axis([0,1,0,2])
plt.title("PMH0")

plt.subplot(3,1,2); 
plt.plot(pmh1.th[:,0],pmh1.th[:,1],'r'); xlabel("th0"); ylabel("th1"); 
plt.hold("on"); 
plt.plot(pmh1.th[:,0],pmh1.th[:,1],'r.');
plt.plot([sys.par[0],sys.par[0]],[0,2],'k:')
plt.plot([0,1],[sys.par[1],sys.par[1]],'k:')
plt.hold("off"); 
plt.axis([0,1,0,2])
plt.title("PMH1")

plt.subplot(3,1,3); 
plt.plot(pmh2.th[:,0],pmh2.th[:,1],'b'); xlabel("th0"); ylabel("th1"); 
plt.hold("on"); 
plt.plot(pmh2.th[:,0],pmh2.th[:,1],'b.');
plt.plot([sys.par[0],sys.par[0]],[0,2],'k:')
plt.plot([0,1],[sys.par[1],sys.par[1]],'k:')
plt.hold("off"); 
plt.axis([0,1,0,2])
plt.title("PMH2")

#############################################################################################################################
# End of file
#############################################################################################################################
