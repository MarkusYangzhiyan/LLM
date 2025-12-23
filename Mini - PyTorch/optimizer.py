import numpy as np 

class Optimizer:
    """  
    1. SGD: momentum 
        wt = wt - lr*vt
            vt = gamma * vt + grad
                where gamma is momentum 
    
    2. Adama 
        wt = wt - lr * mt_hat / (np.sqrt(vt_hat) + eps)
            mt_hat = mt / (1-beta1^t)
            vt_hat = vt / (1-beta2^t)
                mt = beta1 * mt + (1-beta1)*grad
                vt = beta2 * vt + (1-beta2)*grad^2

    common params:  wt, lr , grad
        where grad is input 

    SGD params: momentum 
    Adama params: beta1,beta2
    
    

    """
    def __init__(self,params,lr= 0.01,name = "optimizer"):
        self.params = params
        self.lr = lr


    def step(self,grads):
        raise NotImplementedError
    


class SGD(Optimizer):
    def __init__(self, params, lr=0.01,momentum = 0.9, name="sgd"):
        super().__init__(params, lr, name)
        self.momentum = momentum
        self.vs = [ np.zeros_like(p) for p in self.params ]

        
    def step(self,grads):

        for i,(param,grad) in enumerate(zip(self.params,grads)):
            self.vs[i] = self.momentum * self.vs[i] + grad
            param -= self.lr * self.vs[i]


class Adam(Optimizer):
    def __init__(self, params, lr=0.01,beta1 = 0.9,beta2 = 0.999, eps = 1e-8,name="adam"):
        super().__init__(params, lr, name)

        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.eps = eps

    def step(self,grads):
        self.t = self.t + 1 
        for i, (param,grad) in enumerate(zip(self.params,grads)):
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1)*grad
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2)*(grad**2)

            m_hat = self.m[i] / (1-self.beta1**self.t)
            v_hat = self.v[i] / (1-self.beta2**self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


