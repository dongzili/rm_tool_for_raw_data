from fit_RM_linear import *
import numpy as np
import numpy.ma as ma

testData='velaGBT_4gpu_freqQUVweight_chan.npy'
data=np.load(testData)
freqArr,IQUV=data[0],data[1:]
IQUV=ma.array(IQUV);IQUV[:,IQUV[0]==0]=ma.masked
IQUV*=IQUV[0]

numSubBand=4

fit=FitFaradayLinear(freqArr,numSubBand=numSubBand)
#to set the initial guess for the paramters
tau, rm,psi,phi =(0.136127),31.38,0,0
pInit=np.array([rm,tau,tau,tau,tau,psi,psi,psi,psi,phi])
#default bounds and method for fitting
bounds=(-np.inf,np.inf)
method='trf'
#how to set bounds for the fitting
bounds=([-np.inf,0.,0.,0.,0.,-np.pi,-np.pi,-np.pi,-np.pi,-np.pi],
        [np.inf,0.3,0.3,0.3,0.3,np.pi,np.pi,np.pi,np.pi,np.pi])
#if using non-inf bounds, could use this method
#method='dogbox'
para,paraErr=fit.fit_rm_cable_delay(pInit,IQUV,power2Q=1,bounds=bounds,method=method)
#IQUVerr
fit.show_fitting(para,IQUV[1:],numSubBand=numSubBand,power2Q=1,
        title='fitting with cable delay')

#derotate the cable delay only
#use fit.rot_back_QUV if the QUV array is of shape (3, freq channel)
#use fit.rot_back_QUV_array if the QUV array is of shape (3, freq channel, pulse gates)
QUVcableDelayCorrected=fit.rot_back_QUV(para,IQUV[1:],numSubBand=numSubBand,power2Q=1)
