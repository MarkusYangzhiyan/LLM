import numpy as np 

class Layers:
    def __init__(self,name = 'layers'):

        # 所有子类都需要用到的参数
        self.params = {}
        self.grads = {}
        self.name = name
        self.is_training = True     # 控制training时dropout等行为

    def forward(self,inputs):
        # 子类必须实现，如果子类缺少forward则会抛出警告
        raise NotImplementedError
    
    def backward(self,grad_output):
        # 同上
        raise NotImplementedError
    
    def train(self):
        # 切换到training时，dropout开关开启
        self.is_training = True

    def eval(self):
        # 切换到evaluation时，dropout关闭
        self.is_training = False

