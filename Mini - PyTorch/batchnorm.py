import numpy as np
from layers import Layers

class BatchNorm(Layers):
    def __init__(self, num_features, momentum=0.9, eps=1e-5, name="batchnorm"):
        super().__init__(name)
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # 可学习参数 (Learnable Parameters)
        # Gamma: 缩放因子 (初始化为1)
        # Beta: 平移因子 (初始化为0)
        self.params['gamma'] = np.ones((1, num_features))
        self.params['beta'] = np.zeros((1, num_features))
        
        # 全局统计量 (Running Stats) - 用于测试阶段 (eval)
        # 不需要反向传播更新，而是用滑动平均更新
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        self.cache = None

    def forward(self, x):
        # x shape: (N, D)
        if self.is_training:
            # 1. 计算当前 Batch 的均值和方差
            sample_mean = np.mean(x, axis=0, keepdims=True)
            sample_var = np.var(x, axis=0, keepdims=True)
            
            # 2. 归一化 (x - mu) / sqrt(var + eps)
            self.x_centered = x - sample_mean
            self.std_inv = 1.0 / np.sqrt(sample_var + self.eps)
            x_norm = self.x_centered * self.std_inv
            
            # 3. 缓存用于反向传播
            self.cache = x_norm
            
            # 4. 更新全局统计量 (滑动平均)
            # running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var
            
            out = x_norm
        else:
            # 测试阶段：直接使用全局统计量
            # 这是一个关键考点：测试时不能用当前 Batch 的统计量！
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = x_norm
            
        # 5. 缩放和平移 (Scale and Shift)
        # y = gamma * x_norm + beta
        return self.params['gamma'] * out + self.params['beta']

    def backward(self, grad_output):
        # 只有在训练时才需要反向传播
        # BN 的反向传播公式推导非常复杂，这里使用的是化简后的最终公式
        # 参考文献: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
        
        N, D = grad_output.shape
        x_norm = self.cache
        gamma = self.params['gamma']
        
        # 1. 计算 gamma 和 beta 的梯度
        self.grads['gamma'] = np.sum(grad_output * x_norm, axis=0, keepdims=True)
        self.grads['beta'] = np.sum(grad_output, axis=0, keepdims=True)
        
        # 2. 计算输入 x 的梯度 (链式法则最难的部分)
        # 这里的 dx 公式是经过数学化简的高效版本
        dx_norm = grad_output * gamma
        
        # 公式解释：
        # dL/dx = (1/N*std) * [ N * dL/dx_norm - sum(dL/dx_norm) - x_norm * sum(dL/dx_norm * x_norm) ]
        dx = (1.0 / N) * self.std_inv * (
            N * dx_norm - 
            np.sum(dx_norm, axis=0, keepdims=True) - 
            x_norm * np.sum(dx_norm * x_norm, axis=0, keepdims=True)
        )
        
        return dx