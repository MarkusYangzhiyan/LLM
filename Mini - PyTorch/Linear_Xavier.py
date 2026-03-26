import numpy as np 
from Layer import Layer 

# 注意点
## 记得可以使用Xavier初始化也可以使用He初始化
## 每一个部分的返回值是什么要搞清楚


# 面试题1
## 1. 请现场推导一下W的梯度公式
###     已知 y = XW + b   并且根据反向传播链式法则   dL/dw = dL/dy * dy/dw 
###     假设dL/dy 我们称作是 grad_output 并且维度和y的维度相同是 (N,d_output)
###     那么 dy/dw = X 
###     我们需要求的W的梯度也需要和参数W的维度一致 (d_input,d_output)
###     所以我们需要矩阵的转置变换  (d_input,N) x (N,d_output)
###     结果： self.grads['W'] = X.T @ grad_output

# 面试题2
## 2. 为什么这里使用的是Xavier初始化?  还有其他初始化的方法吗？ 有什么区别？
###     首先除了Xavier初始化外，还有He KaiMing 初始化
###     我们假设activation function是线性的，比如说sigmoid或者Tanh，那么Xavier初始化更适合
###     如果activation fct使用的ReLU，应该使用He初始化  limit = np.sqrt(6.0/d_input)，否则会导致深层网络信号消失
###     为什么呢？
###     深度学习初始化的逻辑是：确保信息在经过多层时，每一层输出的方差保持一致。
###     如果方差随层数增加而越来越大，会导致梯度爆炸,反之会导致梯度消失
###     那么我们需要找一个初始化权重W的方差Var(W)，使得输出y的方差和输入x的方差基本一致
###     xavier初始化的推导逻辑是：y = Σwi*xi  Var(y) = d_input * Var(w) * Var(x)  这里假设是iid且均值为0
###     为了让Var(y) = Var(x),需要Var(w) = 1/d_input,考虑到反向传播，Xavier取输入和输出维度的均值2/(d_input+d_output)
###     但是当使用ReLU的时候，ReLU = max(0,z),它会强行切断一半的负信号，所以此时Xavier初始化就失效了
###     经过ReLU后，输出的方差大致是原来的一半，那么此时为了保证前后方差一致性，我们Var(w)应该是2/d_input
###     如果和Xavier一样的话，前向传播的方差约束是2/d_input,反向传播的方差约束是2/d_output
###     但是论文中指出，不需要像Xavier一样求平均了，只要满足其中任意一个条件，梯度就不会消失或者爆炸，所以只取2/d_input

class Linear(Layer):
    def __init__(self,d_input,d_output,name = 'linear_xavier'):
        super().__init__(name)

        """
        初始化
        W使用Xavier初始化，b设置为0
        X的维度 (batch_size = N, d_input)
        y的维度 (N,d_output)
        W的维度 (d_input,d_output)
        """
        limit = np.sqrt(6.0 / (d_input + d_output))

        self.params['W'] = np.random.uniform(-limit,limit,(d_input,d_output)) 

        self.params['b'] = np.zeros((1,d_output))   # 后续配合广播机制

        self.cache = None

    def forward(self,input):
        "y = XW + b"

        self.cache = input 
        return input @ self.params['W'] + self.params['b']
    
    def backward(self,grad_output):
        """
        here we need 3 items:
            one is self.params['W']     (d_input,d_output)
            two is self.params['b']     (1,d_output)
            three is grad_output for this layer which is also grad_input for next layer   dL/dx  (N,d_input)

        the initial grad_output  ==  yi_hat - yi  which is derivative of loss function L

        grads['W'] = dL/dw = dL/dy * dy/dw = grad_output * X    
            we should adjust the dim of grads['W'] to (d_intput,d_output) 
            so grads['W'] should be the transpose of X(d_input,N) @ derivative of Loss(N,d_output)

        grads['b'] = dL/db = dL/dy * dy/db = grad_output * 1  
            needed to be added up for each row

        grad_output = dL/dx = dL/dy * dy/dx = grad_output * params['W']
            the dim of grad_out is (N,d_input) 
            grad_output (N,d_output) @ params['W'].T (d_output,d_input)

        """

        X = self.cache 
        self.grads['W'] =  X.T @ grad_output 
        self.grads['b'] =  np.sum(grad_output,axis = 0,keepdims=True)

        grad_output = grad_output @ self.params['W'].T

        return grad_output 




        
