from fit_RM_linear import *
import numpy as np
import numpy.ma as ma

testData='velaGBT_4gpu_freqQUVweight_chan.npy'
data=np.load(testData)
freqArr,weight,scaledQUV=data[0],data[1],data[2:]
scaledQUV=ma.masked_values(scaledQUV*weight[np.newaxis,:],0)

numSubBand=4

fit=FitFaradayLinear(freqArr,numSubBand=numSubBand)
tau, rm,phi,psi =(0.136127),31.38,0,0
pInit=np.array([rm,tau,tau,tau,tau,phi,phi,phi,phi,psi])
#pInit=np.array([34.78997907,   0.10591635,   0.13284147 ,  0.13773862,   0.14378319, -2.33632502 , -1.24739466,   2.21203521,  -0.484787,0 ])
pOut=fit.fit_rm_cable_delay(pInit,scaledQUV,power2Q=1)
fit.show_fitting(pOut,scaledQUV,numSubBand=numSubBand,power2Q=1)
