
# coding: utf-8

# In[ ]:

## data generator
import GPy
import numpy as np

class DataGenerator:
    def __init__(self,otherKernel = 'gaussian'):
        self.kernels = self.generateKernels(otherKernel)
    
    def generateKernels(self,otherKernel):
        kernels = []
        kernels.append(GPy.kern.Linear(1,variances=3) + GPy.kern.ExpQuad(1,lengthscale=1,variance=2))
        kernels.append(GPy.kern.Linear(1,variances=3) + GPy.kern.Brownian(1,variance=2))
        if otherKernel == 'cyclic':
            kernels.append(GPy.kern.Cosine(1,lengthscale=4,variance=2))
        else:
            kernels.append(GPy.kern.ExpQuad(1,lengthscale=1,variance=2))
        kernels.append(GPy.kern.ExpQuad(1,lengthscale=1,variance=2))
        return kernels
    
    def generatelatent(self,N,noiselevel=0.05):
        def scaleIt(d):
            return (d - d.min())/(d.max() - d.min())
        
        x = np.linspace(0,N-1,N)
        u1 = np.random.multivariate_normal(mean=np.zeros(N).flatten(),cov=self.kernels[0].K(x.reshape(N,1))) + noiselevel* np.random.normal(0,1,N)
        u2 = np.random.multivariate_normal(mean=np.zeros(N).flatten(),cov=self.kernels[1].K(x.reshape(N,1))) + noiselevel* np.random.normal(0,1,N)
        g1 = np.random.multivariate_normal(mean=np.ones(N).flatten(),cov=self.kernels[2].K(x.reshape(N,1))) + noiselevel* np.random.normal(0,1,N)
        g2 = np.random.multivariate_normal(mean=np.ones(N).flatten(),cov=self.kernels[3].K(x.reshape(N,1))) + noiselevel*np.random.normal(0,1,N)
        return scaleIt(u1),scaleIt(u2),scaleIt(g1),scaleIt(g2)

    
    def generateData(self,phi,P=2,S=100,C=3,N=100):
        #phi = np.random.normal(loc=0,scale=1,size=(C,P))
        phiBar = np.matrix(np.kron(phi,np.eye(N)))
        uBar = np.matrix(np.ones((S,N*P)))
        L = np.zeros(S)
        for s in range(S):
            u1,u2,g1,g2 = self.generatelatent(N=N)
            while u1[0] > u1[-1]: ## to make sure that we have only increasing/cyclic trend 
                u1,u2,g1,g2 = self.generatelatent(N=N)
            if np.random.rand() < 0.4: #Include trend lines
                uBar[s,:] = np.concatenate([u1,g1])
                L[s] = 1
            else:
                uBar[s,:] = np.concatenate([g2,g1])
                L[s] = -1
        print S,C,P,N
        YBar = uBar * phiBar.T + np.random.random()
        return YBar,L,uBar

