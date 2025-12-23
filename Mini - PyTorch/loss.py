import numpy as np 

class MSEloss:
    def __init__(self, name = 'mseloss'):
        self.prediction = None
        self.target = None

        
    def forward(self,prediction,target):

        #  L = (predictions - target) ** 2 / N
        #  dL/dz = 2 * (predictions - target) / N 

        self.prediction = prediction
        self.target = target


        loss = np.mean((prediction - target)**2)

        return loss
    
    def backward(self):

        N = self.prediction.size  

        grad_output = 2 * (self.prediction - self.target) / N 

        return grad_output



class CrossEntropyloss:
    def __init__(self,name = 'crossentropyloss'):
        self.target = None
        self.probs = None
        pass

    def forward(self,logits,target):

        # L = - Σ yi * log(yi_hat)
        # yi_hat = softmax(zi)  where zi is logits
        # softmax = exp/Σexp
        # dL/dz = (y_hat - y) / N
        
        self.target = target
        shifted_logits = logits - np.max(logits, axis = 1, keepdims= True)
        exps = np.exp(shifted_logits)
        self.probs = exps / np.sum(exps,axis = 1,keepdims = True)
        batch_loss = -np.sum(target * np.log(self.probs + 1e-15), axis =1,keepdims=True)
        loss = np.mean(batch_loss)
        return loss

    def backward(self):

        N = self.probs.shape[0]

        return (self.probs - self.target) / N

