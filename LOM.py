
# coding: utf-8

# ### Class structure for SLFM 

# In[1]:

### necessary Imports
import numpy as np
import pylab as pb
import scipy as sc
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as MNormalDist
import GPy

#get_ipython().magic(u'pylab inline')


# In[120]:

from scipy.stats import norm as norm
class LOM():
    
    def __init__(self,Y=None,L=None,N=0,C=0,S=0):
        self.N = N
        self.S = S
        self.Y = Y
        self.L = L
        self.C = C
        self.n = self.N
        self.P = 1
        
        self.fitted = False
    
    def fit(self,n,P=1,iters=500,kernels=[]):
        self.P = P
        self.n = n        
        if len(kernels) == 0:
            kernels = [GPy.kern.ExpQuad(1,lengthscale=0.5) for p in range(self.P)]
        self.setupVI(kernels)
        self.VI(iters)
        self.fitted=True
    
    def guessLabel(self,l,mu=0):
        if l < mu:
            return -1
        else:
            return 1
        
    def calculateTNMean(self,Mu):
        means = np.zeros(self.S)
        for s in range(self.S):
            if self.L[s] == 1:
                a = 0
                b = np.infty
            else:
                a = -np.infty
                b = 0
            if Mu[s,0] < -5:
                Mu[s,0] = -5
            if Mu[s,0] > 5:
                Mu[s,0] = 5
            alpha = a - Mu[s,0]
            beta = b - Mu[s,0]
            Z = norm.cdf(beta) - norm.cdf(alpha)
            #print beta, alpha,Z
            means[s] = Mu[s,0] + (norm.pdf(alpha) - norm.pdf(beta))/Z
        return means

    def setupVI(self,kernels=[]):
        ## get random inducing point
        ## Intialization for variational inference
        noOfInductions = self.n
        x = np.linspace(0,self.N-1,self.N)
        n = np.sort(np.random.choice(x,size=noOfInductions,replace=False))
        nBar = np.concatenate([c*self.N+n for c in range(self.C)])
        NBar = np.concatenate([c*self.N+x for c in range(self.C)])
        # Create Matrices
        self.Knn = list()
        self.KNn = list()
        self.KnnInv = list()
        self.KNN = list()
        ls = [2,2,2,2,2]
        for i in range(self.P):
            kenrel = kernels[i]
            self.Knn.append(kenrel.K(n.reshape(noOfInductions,1)))
            self.KNn.append(kenrel.K(x.reshape(self.N,1), n.reshape(noOfInductions,1)))
            self.KNN.append(kenrel.K(x.reshape(self.N,1)))
            self.KnnInv.append(np.linalg.inv(self.Knn[i]))
        self.Kpnn = np.matrix(sc.linalg.block_diag(*self.Knn))
        self.KpNn = np.matrix(sc.linalg.block_diag(*self.KNn))
        self.KpnnInv = np.matrix(sc.linalg.block_diag(*self.KnnInv))
        self.KpNN = np.matrix(sc.linalg.block_diag(*self.KNN))
        # Generate a guess for phi
        self.phiHatMean = 0
        self.phiHatV = 1
        self.phiHatExp = np.mat(np.random.normal(loc=self.phiHatMean,scale=self.phiHatV,size=(self.C,self.P)))
        self.phiHatExpBar = np.kron(self.phiHatExp,np.eye(self.N))
        self.phiHatVar = np.eye(self.P)
        # Generate a guess for B
        self.BHatMean = 0
        self.BHatV = 1
        self.BHatExp = np.mat(np.random.normal(loc=self.BHatMean,scale=self.BHatV,size=(self.S,1)))
        # Generate a guess for W
        self.WHatMean = 0
        self.WHatV = 1
        self.WHatExp = np.mat(np.random.normal(loc=self.WHatMean,scale=self.WHatV,size=(1,self.P*self.N)))
        self.WHatV = np.eye(self.P*self.N)
        #Generate samples for latent
        self.uHatMeansBar = np.random.normal(loc=0,scale=1,size=(self.S,noOfInductions*self.P,1))
        self.uHatVsBar = np.eye(noOfInductions*self.P)
        self.uExpBar = np.random.normal(loc=0,scale=1,size=(self.S,self.N*self.P))
        print "Concatenated Latent Gaussian Processes:"
        #i=plt.plot(self.uExpBar.T)
        #plt.show()
        print self.KpNn.shape,self.Kpnn.shape,
        ## Generate Sample for latent labels
        self.lHatExp = (self.uExpBar * self.WHatExp.T) + self.BHatExp
        ## Get initial label guess
        self.LHat = np.ones(self.S)
        self.LHat = [self.guessLabel(self.lHatExp[s]) for s in range(self.S)]
        
    #Calculate F and Z
    def calculateVZ(self,uVBar,uExpBar,YBar,P,C,S,N):
        Vphi = np.zeros((P,P))
        Zbar = np.zeros((P,C))
        #SUm over S's
        sigma = np.zeros((P,P))
        for p in range(P):
            sigmaPP = uVBar[p*N:p*N+N,p*N:p*N+N]
            #print p, sigma
            sigma[p,p] = np.trace(sigmaPP)
        for s in range(S):
            usBar = uExpBar[s].reshape(P,N)
            ysBar = YBar[s].reshape(C,N)
            Vphi += usBar*usBar.T + sigma
            Zbar += usBar*ysBar.T
        return sigma,Vphi,Zbar
    
    
    
    
    def VI(self,iters):
        # to speed up calculate M in the beginning
        self.M = np.matrix(self.KpnnInv * self.KpNn.T)
        self.KpNn_nnInv_nN = self.KpNn*self.KpnnInv*self.KpNn.T
        for i in range(iters):
            if i%(iters/10)==0:
                print i,
            ## Update for uHat
            self.phiHatExpBar = np.kron(self.phiHatExp,np.eye(self.N))
            self.phiHatVarBar = np.kron(self.phiHatVar,np.eye(self.N))
            self.Fu = self.phiHatExpBar.T*self.phiHatExpBar + self.phiHatVarBar + self.WHatExp.T*self.WHatExp + self.WHatV
            self.uHatVsBar = np.linalg.inv(self.KpnnInv + self.KpnnInv*self.KpNn.T*self.Fu*self.KpNn*self.KpnnInv)
            self.uHatMeansBar = (self.Y*self.phiHatExpBar + self.lHatExp * self.WHatExp - self.BHatExp * self.WHatExp )* self.KpNn*self.KpnnInv*self.uHatVsBar
            #Update for u 
            #self.M = np.matrix(self.KpnnInv * self.KpNn.T)
            self.uExpBar = self.uHatMeansBar * self.M
            self.uVBar = self.KpNN - self.KpNn_nnInv_nN + self.M.T*self.uHatVsBar*self.M
            #Update for phi
            sigm,VPhi,Zbar = self.calculateVZ(self.uVBar,self.uExpBar,self.Y,self.P,self.C,self.S,self.N)
            self.phiHatVar = np.linalg.inv(VPhi + np.eye(self.P))
            self.phiHatExp = np.matrix(Zbar.T) * self.phiHatVar
            #Update for W
            self.WHatV = np.linalg.inv(self.uExpBar.T*self.uExpBar + self.uVBar + np.eye(self.uVBar.shape[0]))
            self.WHatExp =  (self.lHatExp.T*self.uExpBar - self.BHatExp.T*self.uExpBar )*self.WHatV
            #Update for B
            self.BHatExp = (np.ones(self.S)*np.sum(self.lHatExp - self.uExpBar * self.WHatExp.T)/(self.S+1)).reshape(self.S,1)
            #Update for l
            self.lHatExp = self.calculateTNMean(self.uExpBar*self.WHatExp.T + self.BHatExp).reshape(self.S,1)

    def predict(self,Ytest):
        ltest=np.zeros(Ytest.shape[0])
        Lpred=np.zeros(Ytest.shape[0])
        utestHatMeansBar = (Ytest*self.phiHatExpBar )* self.KpNn*self.KpnnInv*self.uHatVsBar
        #Update for u 
        M = np.matrix(self.KpnnInv * self.KpNn.T)
        utestExpBar = utestHatMeansBar * M
        for t in range(Ytest.shape[0]):
            ltest[t] = np.dot(utestExpBar[t,:] ,self.WHatExp.T) + self.BHatExp[0,0]
            Lpred[t] = self.guessLabel(ltest[t])
        return Lpred,ltest,utestExpBar


