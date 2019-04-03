import numpy as np
from numpy import cos,sin
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
#################################################################
#------------------------------------------------
class FitFaradayLinear:
    '''
    fit RM and cable delay from QUV received with linear receiver
    '''
    def __init__(self,freqarr,numSubBand=1):
        '''
        INPUT:
            an array of data frequency
        OPTIONAL:
            number of sub band, default:1

        assign parameter: freqArr, scaledLambdaSquare, scaledFreqArr
        the latter two array calculated respect to the center of the band,
        used for fitting
        '''
        c=299.792458 #m/s /MHz2Hz light speed
        numChannels=len(freqarr)
        self.numSubBand=numSubBand
        self.freqArr=freqarr
        self.scaledLambdaSquare=(c/self.freqArr)**2\
                -(c/self.freqArr[numChannels//2])**2
        fcen=self.freqArr.reshape(numSubBand,-1).mean(-1,keepdims=True)
        #fcen=np.array([ 798.125,  848.125,  873.125,  898.125])[:,np.newaxis]
        self.scaledFreqArr=(np.copy(self.freqArr).reshape(numSubBand,-1)-fcen).ravel()
        self.noCableDelay=0

    def _test_data_dimension(self,data):
        ''' last dim of the QUV data should be freq'''
        if data.shape[-1]!=len(self.freqArr):
            print('input data dim %d does not match the numChannels specified %d')%(data.shape[-1],len(self.freqArr))
            return -1
        else:
            return 0

    #generate rotation matrix
    def rotV(self,theta):
        R = np.identity(3)
        R[0,0] = cos(theta)
        R[1,1] = cos(theta)
        R[0,1] = -sin(theta)
        R[1,0] = sin(theta)
        return R

    def rotQ(self,theta):
        R = np.identity(3)
        R[1,1] = cos(theta)
        R[2,2] = cos(theta)
        R[1,2] = -sin(theta)
        R[2,1] = sin(theta)
        return R

    def rot_back_matrix(self,theta1,theta2): #rm lambda2, tau f
        return np.dot(self.rotV(theta1),self.rotQ(theta2))


    def rot_back_QUV_array(self,pars,QUVgates,numSubBand=1,power2Q=1):
        '''
        correct the RM and cable delay
        INPUT: pars,QUV (of shape (3, number of gates))
        OUTPUT: the corrected QUV (of shape (3, number of gates))
        '''
        RM=pars[0]; tau=(pars[1:1+numSubBand])
        psi=pars[1+numSubBand:1+numSubBand*2]%(2*np.pi);
        if self.noCableDelay==1:
            tau[:]=0
            psi[:]=0
        if power2Q==1:
                phi=pars[-1] #to rot all U to Q
        #two rotation
        FaradayRot=2*RM*self.scaledLambdaSquare;
        if power2Q==1:
                FaradayRot+=phi #to rot all U to Q

        bandedFreq=np.copy(self.scaledFreqArr).reshape(numSubBand,-1)
        CableRot=tau[:,np.newaxis]*bandedFreq+psi[:,np.newaxis]
        CableRot=CableRot.ravel()

        rottedQUV=np.zeros(QUVgates.shape,dtype='float')
        For k in np.arange(len(self.freqArr)):
            rottedQUV[:,k,:]=np.einsum('ij,jk->ik',self.rot_back_matrix(-FaradayRot[k],-CableRot[k]),QUVgates[:,k,:])
        return rottedQUV

    def rot_back_QUV(self,pars,QUV,numSubBand=1,power2Q=1):
        '''
        correct the RM and cable delay
        INPUT: pars,QUV
        OUTPUT: the corrected QUV
        '''
        RM=pars[0]; tau=(pars[1:1+numSubBand])
        psi=pars[1+numSubBand:1+numSubBand*2]%(2*np.pi);
        if self.noCableDelay==1:
            tau[:]=0
            psi[:]=0
        if power2Q==1:
                phi=pars[-1] #to rot all U to Q
        #two rotation
        FaradayRot=2*RM*self.scaledLambdaSquare;
        if power2Q==1:
                FaradayRot+=phi #to rot all U to Q

        bandedFreq=np.copy(self.scaledFreqArr).reshape(numSubBand,-1)
        CableRot=tau[:,np.newaxis]*bandedFreq+psi[:,np.newaxis]
        CableRot=CableRot.ravel()

        rottedQUV=np.zeros(QUV.shape,dtype='float')
        for k in np.arange(len(self.freqArr)):
            rottedQUV[:,k]=self.rot_back_matrix(-FaradayRot[k],-CableRot[k]).dot(QUV[:,k])
        return rottedQUV

    def _loss_function(self,pars,QUV,weight,power2Q):
        '''the function to minimize during the fitting'''
        rottedQUV=self.rot_back_QUV(pars,QUV,numSubBand=self.numSubBand,power2Q=power2Q)

        avQUV=rottedQUV.mean(-1,keepdims=True)
        #rot all power to Q
        if power2Q==1:
                avQUV[1]=0 #want Q to be close to 0
        distance= (np.abs(rottedQUV-avQUV)*weight).ravel() ##flatten QUV
        #print(distance.sum())
        return distance

    def fit_rm_cable_delay(self,pInit,QUV,maxfev=20000,ftol=1e-3,weight=None,power2Q=0,bounds=(-np.inf,np.inf),method='trf',noCableDelay=0):
        '''fitting RM and cable delay:
        INPUT:
            initial parameter: pInit=np.array([RM,np.repeat(tau,numSubBand),np.repeat(psi,numSubBand),phi])
                               RM: rotation measure
                               tau: cable delay
                               psi: a constant phase btween U and V for different sub band
                               phi: a constant phase between Q and U
                               later two are used to rotate all power to Q
            QUV: 3 by len(freq) array

        OPTIONAL:
            weight: an array with the same length as the input frequency, default weight=1.
            power2Q: whether to rotate all the power in U to Q
            parameters for least_squares:
            maxfev,ftol: parameters for leastsq function, default: maxfev=20000,ftol=1e-3
            bounds:default:(-np.inf,np.inf)
            method: default:'trf'
        '''
        self.noCableDelay=noCableDelay
        if self._test_data_dimension(QUV)!=0:
            return -1
        if weight is None:
            weights=1.
        else:
            weights=np.copy(weight)/weight.mean()

        paramFit = least_squares(self._loss_function,pInit,args=(QUV/QUV.mean(),weights,power2Q),max_nfev=maxfev,ftol=ftol,bounds=bounds,method=method)
        return paramFit

    def show_fitting(self,pars,QUV,I=None,numSubBand=1,power2Q=0,returnPlot=0,fmt='.',title=''):
        '''show QUV matrix before fitting and the QUV after corrected with the fitted parameters
           INPUT:
            pars:the output parameter from fit_rm_cable_delay, it has the same format as pInit
            QUV: the 3 by len(freq) array you used to feed into fit_rm_cable_delay
        '''
        rottedQUV=self.rot_back_QUV(pars,QUV,numSubBand=numSubBand,power2Q=power2Q)
        labels=['Q','U','V']
        fig,axes=plt.subplots(2,1,figsize=[8,8],sharex=True,sharey=True)

        if I is not None:
            for i in np.arange(2):
                axes[i].plot(self.freqArr,I,'k.',label='I')

        for i in np.arange(2):
            for j in np.arange(len(labels)):
                if i==0:
                    axes[i].plot(self.freqArr,QUV[j],fmt,label=labels[j])
                else:
                    axes[i].plot(self.freqArr,rottedQUV[j],fmt,label='rotted '+labels[j])
            axes[i].legend()
            axes[i].axhline(y=0,color='k')
        axes[0].set_title(title)
        if returnPlot==1:
                return fig,axes
        plt.show()

