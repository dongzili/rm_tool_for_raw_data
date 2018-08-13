import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
#################################################################
#------------------------------------------------
class FitFaradayCircular:
    '''
    fit RM and cabel delay
        fitted pars: RM,cable_delay,\
                constant used to rotate all the power to Q (default: 1 constant/subband)
        initial guess: RM:rad/m^2, tau:rad/MHz, and a constant phase
    '''
    def __init__(self,freqarr):
        '''
        assign parameter: freqArr, scaledLambdaSquare, scaledFreqArr 
        the latter two array calculated respect to the center of the band,
        used for fitting
        '''
        numChannels=len(freqarr)
        self.freqArr=freqarr
        self.scaledLambdaSquare=(299.8./self.freqArr)**2\
                -(299.8./self.freqArr[numChannels//2])**2
        self.scaledFreqArr=np.copy(self.freqArr)-self.freqArr[numChannels//2]

    def _test_data_dimension(self,data):
        ''' last dim of the QUV data should be freq'''
        if data.shape[-1]!=len(self.freqArr):
            print('input data dim %d does not match the numChannels specified %d')%(data.shape[-1],len(self.freqArr))
            return -1
        else:
            return 0

    def rot_angle(self,pars):
        '''
        the joint rotated angle due to RM and cable delay
        '''
        RM=pars[0]; tau=pars[1] 
        psi=pars[2:]%(2*np.pi); numSubBand=len(psi)
        rot_angle=2*RM*self.scaledLambdaSquare;
        rot_angle[:]+=((tau*self.scaledFreqArr[:]).reshape(numSubBand,-1)+psi[:,np.newaxis]).reshape(-1)
        return rot_angle

    def _loss_function(self,pars,lr):
        '''the function to minimize during the fitting'''
        lr_rot=np.copy(lr)*np.exp(-1j*self.rot_angle(pars))
        distance=np.concatenate((np.abs(lr_rot.real-np.abs(lr)),np.abs(lr_rot.imag)))
        return distance

    def fit_rm_cable_delay(self,pInit,lr,maxfev=20000,ftol=1):
        '''fitting RM and cable delay:
        INPUT: 
        initial parameter:[RM,tau,psi]
        QUV: 3 by len(freq) array
        '''
        if self._test_data_dimension(lr)!=0:
            return -1
        paramFit = leastsq(self._loss_function,pInit,args=(lr),maxfev=maxfev,ftol=ftol)[0]
        return paramFit

    def derotate(self,pars,lr):
        return lr*np.exp(-1j*self.rot_angle(pars))

    def show_fitting(self,pars,lr):
        lr_fit=np.absolute(lr)*np.exp(1j*self.rot_angle(pars))
        plt.figure()
        plt.plot(self.freqArr,lr.real,'.',label='Q')
        plt.plot(self.freqArr,lr.imag,'.',label='U')
        plt.plot(self.freqArr,lr_fit.real,label='Q fit')
        plt.plot(self.freqArr,lr_fit.imag,label='U fit')
        plt.legend()
        plt.show()

    def test(self):
        freqArr,Q,U=np.load('Source_2014-06-13T06:49:56.379828500_underot.npy')
        self.freqArr=freqArr
        pInit=[-50.,1e-8,0]
        pars=self.fit_rm_cable_delay(pInit,Q)
        print('fitted para',pars)
        self.show_fitting(pars,Q+U*1j)

        
    #################################################################
    #USAGE:
    # assume you have an input array lr(f)=Q+iU,
    #if you have more than one subband, eg 3 sub band, then change the subband number in fit_rm_cable_delay, and set p0=np.array([-46.25,0.005,1,1,1])
    #-----------------------------------------------------
    '''
    
    b_fit = leastsq(fit_rm_cabel_delay, p0, args=(lr,self.scaledLambdaSquare,self.scaledFreqArr), maxfev=20000,ftol=1)[0]
    '''