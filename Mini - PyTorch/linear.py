import numpy as np 
from .layers import Layers

class Linear(Layers):
    def __init__(self,d_input,d_output,name = 'linear'):
        super().__init__(name)

        # 各参数的维度
        # X: N x d-input
        # W: d-input,d-output
        # b: 1,d-output
        # Y: N,d-output


        # Xavier initialization 防止层数变多后梯度消失或者梯度爆炸
        limit = np.sqrt(6.0  / (d_input + d_output))
        self.params['W'] = np.random.uniform(-limit,limit,(d_input,d_output))
        self.params['b'] = np.zeros((1,d_output))
        self.cache = None
        


    def forward(self,inputs):
        # Y =XW + b

        # 保存Input用于反向传播
        self.cache = inputs

        return inputs @ self.params['W'] + self.params['b']
    
    def backward(self,grad_output):
        # dW :(d-input,d-output)     
        # dL/dY: N x d_output
        # dW = X * dL/dY
        # db = sigma(delta)     (N,d_output)
        X = self.cache
        self.grads['W'] = X.T @ grad_output
        self.grads['b'] = np.sum(grad_output,axis = 0,keepdims = True)
        

        # dL/dX: N x d_input
        # grad_output: N x d_output
        # W: d_input,d_output
        # grad_input = grad_output * W.T
        # 这一层的输入 = 上一层的输出
        grad_input =  grad_output @ self.params['W'].T 
        return grad_input
    
    