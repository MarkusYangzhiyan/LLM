import numpy as np 
from layers import Layers

class Dropout(Layers):
    def __init__(self, dropout = 0.5, name='dropout'):
        super().__init__(name)

        self.dropout = dropout
        self.mask = None

    def forward(self,inputs):
        if self.is_training:
            # inverted dropout is better than normal dropout, 普通的dropout在丢弃40%后，只剩下60%,那么在测试时，需要改变模型的权重，以便和训练时的大小一致。而inverted dropout在丢弃40%后，让剩余的60%除以了 60%，总参数不变
            # 找一个mask
            # 维度和input相同
            # 超过dropout的部分变成0 
            # 最后分母缩放到1-dropout 

            random_probs = np.random.rand(*inputs.shape)
            keep_mask = random_probs > self.dropout
            scale_factor = 1 / (1 - self.dropout)
            self.mask = keep_mask * scale_factor
            return inputs * self.mask
        else:
            return inputs
        
    def backward(self,grad_output):
        if self.is_training:
            return grad_output * self.mask
        else:
            return grad_output