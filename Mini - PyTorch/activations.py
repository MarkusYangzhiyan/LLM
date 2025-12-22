import numpy as np 
from layers import Layers

class Sigmoid(Layers):
    def __init__(self,name= 'sigmoid'):
        super().__init__(name)

        # fx = 1 / (1 - e^-x)
        # fx' = fx(1-fx)

    def forward(self,inputs):
        self.output = 1 / (1-np.exp(-inputs))
        return self.output
    
    def backward(self,grad_output):
        
        return grad_output * self.output * (1 - self.output)
    

class ReLU(Layers):
    def __init__(self, name='relu'):
        super().__init__(name)
        self.cache = None
        # fx = max(0,x)
        # fx' = 1  if x>0
        #     = 0  if x<=0

    def forward(self,inputs):
        self.cache = inputs
        return np.maximum(0,inputs)
    
    def backward(self,grad_output):
        X = self.cache
        return grad_output * (X > 0)
    

class Tanh(Layers):
    def __init__(self, name='tanh'):
        super().__init__(name)

        # fx =  (e^x - e^-x) / (e^x + e^-x)
        # fx' = 1 - fx^2

    def forward(self,inputs):
        self.output = np.tanh(inputs)
        return self.output
    
    def backward(self,grad_output):
        return grad_output * (1 - self.output **2)
