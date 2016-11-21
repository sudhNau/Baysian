
### necessary Imports
import numpy as np
import pylab as pb
import scipy as sc
from scipy.stats import multivariate_normal as MNormalDist
import GPy
from scipy.stats import norm

## get random inducing point
## Intialization for variational inference
class LCMGP():

    def guessLabel(self,l,mu=0):
        if l < mu:
            return -1
        else:
            return 1

    def initializeVI(self):
        S = self.S
        N = self.N
        C = self.C
        P = self.P
        induction = self.induction
        kernels = self.kernels
        x = np.linspace(0,N-1,N)
        noOfInductions = int(N*induction)
        n = np.sort(np.random.choice(x,size=noOfInductions,replace=False))
        nBar = np.concatenate([c*N+n for c in range(C)])
        NBar = np.concatenate([c*N+x for c in range(C)])
        # Create Matrices
        Knn = list()
        KNn = list()
        KnnInv = list()
        KNN = list()
        ls = [2,2,2,2,2]
        for i in range(P):
            if i < len(kernels):
                kernel = kernels[i]
            else:
                kenrel = GPy.kern.ExpQuad(1,lengthscale=ls[i]) #+ kenrel2
            Knn.append(kenrel.K(n.reshape(noOfInductions,1))+np.eye(noOfInductions)*0.0)
            KNn.append(kenrel.K(x.reshape(N,1), n.reshape(noOfInductions,1)))
            KNN.append(kenrel.K(x.reshape(N,1))+np.eye(N)*0.00)
            KnnInv.append(np.linalg.inv(Knn[i]))
        self.Kpnn = np.matrix(sc.linalg.block_diag(*Knn))
        self.KpNn = np.matrix(sc.linalg.block_diag(*KNn))
        self.KpnnInv = np.matrix(sc.linalg.block_diag(*KnnInv))
        self.KpNN = np.matrix(sc.linalg.block_diag(*KNN))
        # Generate a guess for phi
        phiHatMean = 0
        phiHatV = 1
        self.phiHatExp = np.mat(np.random.normal(loc=phiHatMean,scale=phiHatV,size=(C,P)))
        self.phiHatExpBar = np.kron(self.phiHatExp,np.eye(N))
        self.phiHatVar = np.eye(P)
        # Generate a guess for B
        self.BHatMean = 0
        self.BHatV = 1
        self.BHatExp = np.mat(np.random.normal(loc=self.BHatMean,scale=self.BHatV,size=(S,1)))
        # Generate a guess for W
        self.WHatMean = 0
        self.WHatV = 1
        self.WHatExp = np.mat(np.random.normal(loc=self.WHatMean,scale=self.WHatV,size=(1,P*N)))
        self.WHatV = np.eye(P*N)
        #Generate samples for latent
        self.uHatMeansBar = np.random.normal(loc=0,scale=1,size=(S,noOfInductions*P,1))
        self.uHatVsBar = np.eye(noOfInductions*P)
        self.uExpBar = np.random.normal(loc=0,scale=1,size=(S,N*P))
        #print "Concatenated Latent Gaussian Processes:"
        #i=plt.plot(uExpBar.T)
        #print uHatVsBar.shape,uHatMeansBar.shape
        ## Generate Sample for latent labels
        self.lHatExp = (self.uExpBar * self.WHatExp.T) + self.BHatExp
        self.n=n
        ## Get initial label guess

    def fit(self,Y,L,N,P=3,induction=0.8,iterations=200,kernels=[]):
        self.modelFitted = False
        self.N = N
        self.C = Y.shape[1]/self.N
        self.S = Y.shape[0]
        self.P=P
        self.L = L
        self.YBar=Y
        self.induction = induction
        self.kernels = kernels
        self.initializeVI()
        print "Model initialized with ",induction,"induction ratio"
        print iterations,"updates"
        self.VI(iterations)
        self.modelFitted = True

    def predict(self,Ytest):
        if self.modelFitted == False:
            print "Fit model"
            return
        utestHatMeansBar = (Ytest*self.phiHatExpBar)* self.KpNn*self.KpnnInv*self.uHatVsBar
        #Update for u 
        M = np.matrix(self.KpnnInv * self.KpNn.T)
        utestExpBar = utestHatMeansBar * M
        # Predictions
        ltest=np.zeros(Ytest.shape[0])
        Lpred=np.zeros(Ytest.shape[0])
        for t in range(Ytest.shape[0]):
            ltest[t] = np.dot(utestExpBar[t,:] ,self.WHatExp.T) + self.BHatExp[0,0]
            Lpred[t] = self.guessLabel(ltest[t],self.BHatExp[0,0])
            
        return Lpred,ltest,utestExpBar

    def calculateTNMean(self,Mu,s=-1):
        S = self.S

        means = np.zeros(S)
        for s in range(S):
            if self.L[s] == 1:
                a = 0
                b = np.infty
            else:
                a = -np.infty
                b = 0
            if Mu[0,s] < -5:
                Mu[0,s] = -5
            if Mu[0,s] > 5:
                Mu[0,s] = 5
            alpha = a - Mu[0,s]
            beta = b - Mu[0,s]
            Z = sc.stats.norm.cdf(beta) - sc.stats.norm.cdf(alpha)
            #print beta, alpha,Z
            means[s] = Mu[0,s] + (sc.stats.norm.pdf(alpha) - sc.stats.norm.pdf(beta))/Z
        return means    

    #Calculate F and Z
    def calculateVZ(self,uVBar):
        P=self.P
        N=self.N
        C=self.C
        S=self.S
        Vphi = np.zeros((P,P))
        Zbar = np.zeros((P,C))
        #SUm over S's
        sigma = np.zeros((P,P))
        for p in range(P):
            sigmaPP = uVBar[p*N:p*N+N,p*N:p*N+N]
            #print p, sigma
            sigma[p,p] = np.trace(sigmaPP)
        for s in range(S):
            usBar = self.uExpBar[s].reshape(P,N)
            ysBar = self.YBar[s].reshape(C,N)
            Vphi += usBar*usBar.T + sigma
            Zbar += usBar*ysBar.T
        return sigma,Vphi,Zbar

    def VI(self,iterations=200):
        for i in range(iterations):
            ## Update for uHat
            self.phiHatExpBar = np.kron(self.phiHatExp,np.eye(self.N))
            self.phiHatVarBar = np.kron(self.phiHatVar,np.eye(self.N))
            Fu = self.phiHatExpBar.T*self.phiHatExpBar + self.phiHatVarBar + self.WHatExp.T*self.WHatExp + self.WHatV
            self.uHatVsBar = np.linalg.inv(self.KpnnInv + self.KpnnInv*self.KpNn.T*Fu*self.KpNn*self.KpnnInv)
            self.uHatMeansBar = (self.YBar*self.phiHatExpBar + self.lHatExp * self.WHatExp - self.BHatExp * self.WHatExp )* self.KpNn*self.KpnnInv*self.uHatVsBar
            #Update for u 
            M = np.matrix(self.KpnnInv * self.KpNn.T)
            self.uExpBar = self.uHatMeansBar * M
            self.uVBar = self.KpNN - self.KpNn*self.KpnnInv*self.KpNn.T + M.T*self.uHatVsBar*M
            #Update for phi
            sigm,VPhi,Zbar = self.calculateVZ(self.uVBar)
            self.phiHatVar = np.linalg.inv(VPhi + np.eye(self.P))
            self.phiHatExp = np.matrix(Zbar.T) * self.phiHatVar
            #Update for W
            self.WHatV = np.linalg.inv(self.uExpBar.T*self.uExpBar + self.uVBar + np.eye(self.uVBar.shape[0]))
            self.WHatExp =  (self.lHatExp.T*self.uExpBar - self.BHatExp.T*self.uExpBar )*self.WHatV
            #Update for B
            self.BHatExp = (np.ones(self.S)*np.sum(self.lHatExp - self.uExpBar * self.WHatExp.T)/(self.S+1)).reshape(self.S,1)
            #Update for l
            self.lHatExp = self.calculateTNMean(self.WHatExp*self.uExpBar.T + self.BHatExp).reshape(self.S,1)


# In[ ]:



