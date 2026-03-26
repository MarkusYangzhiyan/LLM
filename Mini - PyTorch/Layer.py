import numpy as np 

# 注意点
## 1. 类名使用Layer更好，单数更好
## 2. 最好加上一个 __call__ 来模拟pytorch的函数调用  例如: forward(input)
## 3. 在forward中最好可以增加一个self.input = input 用来全局储存input用于反向传播

# 面试题1
## 1. 为什么要设置 is_training 这个参数变量？ 
###     因为推理时和训练时是不同的，主要体现在dropout层，如果用的是BN的话，BN在推理和训练时也不同
###     训练时，dropout需要随机丢弃神经元并进行缩放来做正则化避免过拟合，而推理时则保留全部神经元
###     训练时，使用当前的batch计算均值和方差，并更新全局运行均值和方差，但是在推理时是直接使用训练好的全局均值和方差

# 面试题2
## 2. 为什么把params和grads设置成dict的格式？ 有什么好处？
###     在使用Adam优化的时候，优化器可以直接迭代layer.params.keys()，确保每个参数都能与其对应的梯度grads[key]对应
class Layer:
    def __init__(self,name = 'layer'):
        self.name = name 
        self.params = {}
        self.grads = {}
        self.is_training = True
        self.input = None


    def forward(self,input):
        self.input = input 
        raise NotImplementedError
    

    def backward(self,grad_output):
        raise NotImplementedError 
    
    
    def eval(self):
        self.is_training = False 

    def train(self):
        self.is_training = True 

    def __call__(self,input):
        return self.forward(input)
    


