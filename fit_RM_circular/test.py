import numpy as np
from fit_RM_circular import *
fname='Source_2014-06-15T07:28:21.105210250_underot.npy'
freqArr,Q,U=np.load(fname)
freqArr=freqArr.reshape(-1,2).mean(-1)
Q=Q.reshape(-1,2).mean(-1)
U=U.reshape(-1,2).mean(-1)
I=(Q**2+U**2)**0.5
pInit=[-50.,1e-8,0]
#pInit=[-50.,0]
numSubBand=1
fit=FitFaradayCircular(freqArr)

pars=fit.fit_rm_cable_delay(pInit,Q+U*1j,noCableDelay=0,weight=I)
print(pars)
pars=pars.x
print('fitted para',pars)

#using the par to de-rot
lr_derot=fit.derotate(pars,Q+U*1j)

#show fitting
fit.show_fitting(pars,Q+U*1j,I)
