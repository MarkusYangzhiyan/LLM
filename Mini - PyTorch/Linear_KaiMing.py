import numpy as np 
from Layer import Layer

class Linear(Layer):
    def __init__(self,d_input,d_output,name = 'linear'):

        limit = np.sqrt(6.0 / d_input)
        self.params['W'] = np.random.uniform(-limit,limit,(d_input,d_output))
        self.params['b'] = np.zeros((1,d_output))
        self.cache  = None



    def forward(self,input):
        self.cache = input 

        return input @ self.params['W'] + self.params['b']
    
    def backward(self,grad_output):

        X = self.cache 

        self.grads['W'] = X.T @ grad_output 
        self.grads['b'] = np.sum(grad_output,axis=0,keepdims=True)
        grad_output = grad_output @ self.params['W'].T

        return grad_output 

