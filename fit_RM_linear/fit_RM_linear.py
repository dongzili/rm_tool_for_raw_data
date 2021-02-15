## optimize U, with only Q and V, cable delay without phase

import numpy as np
import numpy.ma as ma
from numpy import cos,sin
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from numpy import newaxis as nax

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
        ''' last dim of the IQUV data should be freq'''
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
        INPUT: pars,QUV (of shape (3, chan, number of gates))
        OUTPUT: the corrected QUV (of shape (3, chan, number of gates))
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

        bandedFreq=np.copy(self.freqArr).reshape(numSubBand,-1)
        CableRot=tau[:,np.newaxis]*bandedFreq+psi[:,np.newaxis]
        CableRot=CableRot.ravel()

        rottedQUV=np.zeros(QUVgates.shape,dtype='float')
        for k in np.arange(len(self.freqArr)):
            rottedQUV[:,k,:]=np.einsum('ij,jk->ik',self.rot_back_matrix(-FaradayRot[k],-CableRot[k]),QUVgates[:,k,:])
        return rottedQUV

    def rot_back_QUV(self,pars,QUV,numSubBand=1,power2Q=1):
        '''
        correct the RM and cable delay
        INPUT: pars,QUV (3, chan)
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

        bandedFreq=np.copy(self.freqArr).reshape(numSubBand,-1)
        CableRot=tau[:,np.newaxis]*bandedFreq+psi[:,np.newaxis]
        CableRot=CableRot.ravel()

        rottedQUV=np.zeros(QUV.shape,dtype='float')
        for k in np.arange(len(self.freqArr)):
            rottedQUV[:,k]=self.rot_back_matrix(-FaradayRot[k],-CableRot[k]).dot(QUV[:,k])
        return rottedQUV

    def blackman_smooth(self,I,weightsI=None,smWidth=3.):
            freqReso=np.abs(np.diff(self.freqArr[:2]))
            half=np.array([0.297,0.703])
            wdBin=smWidth/(half[1]-half[0])/freqReso

            window=np.blackman(wdBin)
            window/=window.sum()
            if weightsI is None:
                Ismt=np.convolve(I,window,mode='same')
            else: 
                Ismt=np.convolve(I*weightsI,window,mode='same')
                renorm=np.convolve(weightsI,window,mode='same')
                renorm[renorm==0]=1e5
                Ismt/=renorm
                
            return Ismt

    def _loss_function(self,pars,QUV,weight,power2Q,IsmtRnm):
        '''the function to minimize during the fitting'''
        rottedQUV=self.rot_back_QUV(pars,QUV,numSubBand=self.numSubBand,power2Q=power2Q)

        avQUV=rottedQUV.mean(-1,keepdims=True)*IsmtRnm[nax,:]
        #rot all power to Q
        if power2Q==1:
                avQUV[0]=0 #want Q to be close to 0
        if self.noCableDelay==1:
            pol=[0,1]
        elif self.noCableDelay==2:
            pol=[2]
        else:
            pol=[0,1,2]
        if weight is None:
            distance= (np.abs(rottedQUV[pol]-avQUV[pol])).ravel() ##flatten QUV
        else:
            distance= (np.abs(rottedQUV[pol]-avQUV[pol])*weight[pol]).ravel() ##flatten QUV
        return distance

    def fit_rm_cable_delay(self,pInit,IQUV,maxfev=20000,ftol=1e-3,IQUVerr=None,power2Q=0,bounds=(-np.inf,np.inf),method='trf',noCableDelay=0,smWidth=3.,weights=None):
        '''fitting RM and cable delay:
        INPUT:
            initial parameter: pInit=np.array([RM,np.repeat(tau,numSubBand),np.repeat(psi,numSubBand),phi])
                               RM: rotation measure
                               tau: cable delay
                               psi: a constant phase btween U and V for different sub band
                               phi: a constant phase between Q and U
                               later two are used to rotate all power to Q
            IQUV: 4 by len(freq) array

        OPTIONAL:
            weight: an array with the same length as the input frequency, default weight=1.
            power2Q: whether to rotate all the power in U to Q
            parameters for least_squares:
            maxfev,ftol: parameters for leastsq function, default: maxfev=20000,ftol=1e-3
            bounds:default:(-np.inf,np.inf)
            method: default:'trf'
        '''
        self.noCableDelay=noCableDelay
        if self._test_data_dimension(IQUV)!=0:
            return -1
        if IQUVerr is None:
            weightsI=None
            weightsQUV=None
        else:
            weight=1./IQUVerr
            weight=ma.masked_invalid(weight)
            weight.set_fill_value(0)
            weights=ma.copy(weight)/weight.std()
            weightsI,weightsQUV=weights[0]**2,weights[1:]

        I,QUV=ma.copy(IQUV[0]),ma.copy(IQUV[1:])
        Ismt=self.blackman_smooth(I,weightsI=weightsI,smWidth=smWidth)
        IsmtRnm=Ismt/Ismt.mean()
        
        if weights is not None:
            weightsQUV=np.repeat(weights[None,:],3,axis=0)

        paramFit = least_squares(self._loss_function,pInit,args=(QUV,weightsQUV,power2Q,IsmtRnm),max_nfev=maxfev,ftol=ftol,bounds=bounds,method=method)      
        para,jac=paramFit.x,paramFit.jac
        rottedQUV=self.rot_back_QUV(para,QUV,numSubBand=self.numSubBand,power2Q=power2Q)
        #return para,jac
        jac=jac[:,[0,1,-1]]
        if power2Q==1 and rottedQUV[1].mean()<0:
            para[-1]=(para[-1]+np.pi)%(2*np.pi)
        if noCableDelay==1:
            #para=para[[0,-1]]
            jac=jac[:,[0,-1]]
        if noCableDelay==2:
            #para=para[[1]]
            jac=jac[:,[1]]          
        cov = np.linalg.inv(jac.T.dot(jac))
        paraErr = np.sqrt(np.diagonal(cov))
        print('fitting results para, err',para,paraErr)
        return para,paraErr

    def show_fitting(self,fitPars,QUV,\
                     I=None,numSubBand=1,power2Q=0,returnPlot=0,fmt='.',title='',xlim=None,pol=[0,1,2],fBin=1):
        '''show QUV matrix before fitting and the QUV after corrected with the fitted parameters
           INPUT:
            pars:the output parameter from fit_rm_cable_delay, it has the same format as pInit
            QUV: the 3 by len(freq) array you used to feed into fit_rm_cable_delay
        '''
        if self.noCableDelay==1:
            pars=np.zeros(2+numSubBand*2)
            pars[0],pars[-1]=fitPars[0],fitPars[-1]
        else:
            pars=fitPars
        rottedQUV=self.rot_back_QUV(pars,QUV,numSubBand=numSubBand,power2Q=power2Q)
        labels=['Q','U','V']
        fig,axes=plt.subplots(2,1,figsize=[14,8],sharex=True,sharey=True)

        if I is not None:
            for i in np.arange(2):
                axes[i].plot(self.freqArr.reshape(-1,fBin).mean(-1),I.reshape(-1,fBin).mean(-1),fmt,color='k',label='I')

        for i in np.arange(2):
            for j in pol:
                if i==0:
                    axes[i].plot(self.freqArr.reshape(-1,fBin).mean(-1),QUV[j].reshape(-1,fBin).mean(-1),fmt,label=labels[j])
                else:
                    axes[i].plot(self.freqArr.reshape(-1,fBin).mean(-1),rottedQUV[j].reshape(-1,fBin).mean(-1),fmt,label='rotted '+labels[j])
            axes[i].legend()
            axes[i].axhline(y=0,color='k')
        axes[0].set_title(title)
        if xlim is not None:
            axes[0].set_xlim(xlim)
        if returnPlot==1:
                return fig,axes
        plt.show()

    def show_derotated(self,fitPars,QUV,\
                     I=None,numSubBand=1,power2Q=0,returnPlot=0,fmt='.',title='',xlim=None,pol=[0,1,2],fBin=1):
        '''show QUV matrix before fitting and the QUV after corrected with the fitted parameters
           INPUT:
            pars:the output parameter from fit_rm_cable_delay, it has the same format as pInit
            QUV: the 3 by len(freq) array you used to feed into fit_rm_cable_delay
        '''
        if self.noCableDelay==1:
            pars=np.zeros(2+numSubBand*2)
            pars[0],pars[-1]=fitPars[0],fitPars[-1]
        else:
            pars=fitPars
        rottedQUV=self.rot_back_QUV(pars,np.copy(QUV),numSubBand=numSubBand,power2Q=power2Q)
        labels=['Q','U','V']
        fig,axes=plt.subplots(1,1,figsize=[14,4],sharex=True,sharey=True)

        if I is not None:
            axes.plot(self.freqArr.reshape(-1,fBin).mean(-1),I.reshape(-1,fBin).mean(-1),fmt,color='k',label='I')

        for j in pol:
                    axes.plot(self.freqArr.reshape(-1,fBin).mean(-1),rottedQUV[j].reshape(-1,fBin).mean(-1),fmt,label='rotted '+labels[j])
        axes.legend()
        axes.axhline(y=0,color='k')
        axes.set_title(title)
        if xlim is not None:
            axes.set_xlim(xlim)
        if returnPlot==1:
                return fig,axes
        plt.show()

