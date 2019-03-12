from fit_RM_linear import *
import numpy as np
import numpy.ma as ma

testData='velaGBT_4gpu_freqQUVweight_chan.npy'
data=np.load(testData)
freqArr,I,QUV=data[0],data[1],data[2:]
QUV=ma.array(QUV);QUV[:,I==0]=ma.masked

numSubBand=4

fit=FitFaradayLinear(freqArr,numSubBand=numSubBand)
#to set the initial guess for the paramters
tau, rm,psi,phi =(0.136127),31.38,0,0
pInit=np.array([rm,tau,tau,tau,tau,psi,psi,psi,psi,phi])
#default bounds and method for fitting
bounds=(-np.inf,np.inf)
method='trf'
#how to set bounds for the fitting
bounds=([-np.inf,0,0,0,0,-np.pi,-np.pi,-np.pi,-np.pi,-np.pi],
        [np.inf,0.3,0.3,0.3,0.3,np.pi,np.pi,np.pi,np.pi,np.pi])
method='trf'
#if using non-inf bounds, could use this method
#method='dogbox'
pOut=fit.fit_rm_cable_delay(pInit,QUV,power2Q=1,bounds=bounds,weight=I,method=method)
print('initial fitting',pOut.x,pOut.fun)
fit.show_fitting(pOut.x,QUV,I=I,numSubBand=numSubBand,power2Q=1,
        title='fitting with cable delay')

#derotate the cable delay only
pars=pOut.x
pars[0]=0;pars[-1]=0 #do not correct RM and polarization angle
print('cable delay',pars)
QUVcableDelayCorrected=fit.rot_back_QUV(pars,QUV,numSubBand=numSubBand,power2Q=1)
pOut=fit.fit_rm_cable_delay(pInit,QUV,power2Q=1,bounds=bounds,weight=I,method=method,noCableDelay=1)
pars=pOut.x
print('fitting without cable delay',pars)
#pars[1:-1]=0
fit.show_fitting(pars,QUVcableDelayCorrected,I=I,numSubBand=numSubBand,power2Q=1, 
        title='fitting after correct cable delay')
