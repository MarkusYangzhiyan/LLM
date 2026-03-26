import numpy as np 
from Layer import Layer




# 面试题1
## 1. ReLU有什么优缺点？ 如何解决？
###     优点：ReLU解决了sigmoid作为activation fct会让深层神经网络出现梯度消失的问题
###             sigmoid的导数是fx(1-fx)，其最大值是0.25，根据链式法则连乘的原理，随着层数增加，会出现梯度消失
###             而ReLU下，只要神经元处于激活状态，导数就为1，进行无损反向传播，解决了sigmoid导致梯度消失的问题
###     缺点：但是也有缺点，如果初始化不好，或者学习率太大导致权重更新过猛的话
###             某个神经元可能在所以样本上的输出都是负数，这样的话反向传播梯度就为0了，这个神经元的权重无法更新
###             这就是俗称的 Dead ReLU
###           并且如果ReLU的输出都是非负数的话，下一层神经网络接收的输入均值就大于0了，非零中心化拖慢网络收敛速度
###             这就是俗称的均值偏移 mean shift
###     解决方法： Leaky ReLU
###           fx = x if x>0 else alpha * x 给负值一个微小的斜率alpha = 0.01
###           这样的话即使落入负半区，依然会有微小的梯度传回，解决了Dead ReLU的问题
###     但是： Leaky ReLU也有问题，就是这个hyper parameter alpha的值设置多少才是最好的？
###             引出了PReLU，让神经网络自己去学习和更新这个参数alpha
###             把alpha变成一个参数，通过反向传播更新，让数据自己去决定在负半轴应该保留多少信息，模型性能显著提升。


# 面试题2
## 2. GeLU的原理是什么？ 有什么优缺点？
###     fx = x * Φ(x)   where Φ(x) is the CDF of gaussion distribution
###     因为计算复杂，用tanh来近似 = 0.5 * x * (1 + tanh(  sqrt(2/pi) * (x + 0.044715 * x^3))  )
###     输入的值x乘以一个概率，这个概率取决于x在正态分布中比多少其他输入大。x越大，保留信息越多，反之越少
###     优点： 处处连续可导，极其平滑。不仅能像 Leaky ReLU 一样保留少许负面信息，还能起到类似正则化的作用。
###     缺点： 计算比较复杂，所以在工程中，我们通常用 Tanh 来做近似。
###     解决方法：因为GeLU的计算比较复杂，那么可以使用一个和GeLU形状几乎一样，但是计算更简单的swish来替代它
###                 fx = x * sigmoid(x)  也就是输入x直接乘以它的sigmoid激活函数的值
###                 那么它的有点就是计算比GeLU简单，求导简单，但是缺点就是还是要算一个sigmoid，计算速度慢
###     更好方法：现在的LLM时代，主流的是用SwiGLU
###                 它本质上不再是一个单纯的激活函数，而是一个带有乘法门控机制的网络块。
###                 传统的FFN = W2(Act(W1X+b1)) + b2
###              而SwiGLU将输入x投影到2个不同的空间，一个空间用SiLU激活，另一个空间保持线性关系
###                 然后然后将两者逐元素相乘，最后再投影回来
###                 SwiGLU(x) = ( SiLU(XW1) inner_product (XW2) ) * W3
###         优点： 乘法门控机制引入了极强的非线性表达能力，模型可以通过其中一路来动态“控制”另一路的信息流。
###                  在相同计算量下，SwiGLU 的指标全面碾压传统 FFN。
###         缺点： 参数量增加了。原来只需 2 个权重矩阵，现在需要 3 个。
###                  为了保持参数总量不变，通常需要把隐藏层的维度缩小为原来的 2/3。


class Sigmoid(Layer):
    def __init__(self,name = 'sigmoid'):
        super().__init__(name)


    def forward(self,input):

        """
        fx = 1 / (1+exp(-x))
        the derivative of sigmoid is  fx(1-fx)
        """

        self.output = 1 / (1+np.exp(-input))
        return self.output 
    
    def backward(self,grad_output):
        
        return grad_output * self.output * (1 - self.output)
    

class Tanh(Layer):
    def __init__(self,name = 'tanh'):
        super().__init__(name)


    def forward(self,input):

        """
        fx =  (e^x - e^-x) / (e^x + e^-x)
        fx' = 1 - fx^2
        """

        # p1 = np.exp(input)
        # p2 = np.exp(-input)
        # self.output = (p1-p2)/(p1+p2)

        self.output = np.tanh(input)
        return self.output 
    
    def backward(self,grad_output):

        return  grad_output * (1 - self.output ** 2)
    
class ReLU(Layer):
    def __init__(self,name = 'relu'):
        super().__init__(name)


    def forward(self,input):
        """
        fx = max(0,x)
        fx' = 1 if x > 0 else 0 
        x as input needed in both forward and backward, need a cache to save it 
        """
        self.cache = input 
        return np.maximum(0,input)
    
    def backward(self,grad_output):
        X = self.cache 
        return grad_output * (X > 0)
    

class LeakyReLU(Layer):
    def __init__(self,alpha = 0.01,name = 'leakyrelu'):
        super().__init__(name)

        self.alpha = alpha 
        self.mask = None
        
    def forward(self,input):
        """
        fx = x if x > 0 else alpha * x 
        fx' = 1 if x > 0 else alpha 
        """

        self.mask = (input > 0)

        return np.where(self.mask,input,self.alpha*input) 

    def backward(self,grad_output):

        grad_input = grad_output.copy()
        grad_input[~self.mask] *= self.alpha 


class PReLU(Layer):
    def __init__(self,init_alpha = 0.25,name = 'prelu'):
        super().__init__(name)

        self.params['alpha'] = np.array([init_alpha],dtype=np.float32)
        self.grads['alpha'] = np.zeros(1)
        self.cache = None

    def forward(self,input):

        """
        fx = x if x > 0 else alpha * x 
        fx' = 1 if x > 0 else alpha 
        """

        self.cache = input 
        alpha = self.params['alpha']
        return np.where(input > 0 ,input,alpha * input )
    

    def backward(self,grad_output):
        X = self.cache 
        alpha = self.params['alpha']

        grad_input = grad_output.copy()
        grad_input[X<=0] *= alpha 

        grad_alpha = np.sum(grad_output[X<=0] * X[X<=0])
        self.grads['alpha'] = np.array([grad_alpha])

        return grad_input 



class GeLU(Layer):
    def __init__(self,name = 'gelu'):
        super().__init__(name)
        self.cache = None


    def forward(self,input):
        """
        fx = x * phi(x)

        fx的近似公式 = 0.5 * x * (1 + tanh(  sqrt(2/pi) * (x + 0.044715 * x^3))  )

        inner = np.sqrt(2/np.pi) * (x + 0.044715x^3)

        fx = 0.5 * x * (1 + tanh(inner))

        x : input 
        0.5 * (1 + tanh(inner)) = cdf_approx
        """

        self.cache = input 
        self.inner  = np.sqrt(2/np.pi) * (input + 0.044715 * input ** 3)
        self.cdf_approx = 0.5 * (1.0 + np.tanh(self.inner))
        return input * self.cdf_approx



    def backward(self,grad_output):

        """
        how to calculate fx'?
        fx = x * C(x) where C(x) = 0.5 * (1 + tanh(inner))
        fx' = 1*C(x) + x * C(x)'  where C(x) is cdf_approx and C(x)' is unkonwn and needed to be calculated

        how to calculate C(x)'?
        C(x)' = ( 1 - tanh(inner) ^ 2 ) * inner'
        inner' = np.sqrt(2/pi) * (1 + 0.044715 * 3 * x**2) 
        """
        X = self.cache 
        sech2 = 1.0 - np.tanh(self.inner)**2
        d_inner_dx = np.sqrt(2 / np.pi) * (1.0 + 3 * 0.044715 * X**2)
        d_cdf_dx = 0.5 * sech2 * d_inner_dx
        
        grad_input = self.cdf_approx + X * d_cdf_dx


class SiLU(Layer):
    def __init__(self,name = 'silu'):
        super().__init__(name)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def forward(self,input):
        """
        fx = x * sigmoid(x)
        fx' =  sigmoid(x) + x * sigmoid(x)'
        """

        self.cache = input 
        self.sig_x = self.sigmoid(input)

        self.output = input * self.sig_x
        return self.output 
    
    def backward(self,grad_output):
        X = self.cache 
        
        grad_input = self.sig_x + X * self.sig_x * (1 - self.sig_x)
        
        return grad_output * grad_input
    

class SwiGLU(Layer):
    def __init__(self,name = 'swiglu'):
        super().__init__(name)

    def _sigmoid(self,x):
        return 1 / (1+np.exp(-x))
    
    def forward(self,Y1,Y2):
        """
        fx = ( SiLU(W1X) inner_product W2X ) * W3
        where W1X = Y1 and W2X = Y2
        通过W3映射回来的是FFN线性层，这里只写前面的部分
        """
        self.Y1 = Y1 
        self.Y2 = Y2 
        self.sig_Y1 = self._sigmoid(Y1)
        self.silu_Y1 = Y1 * self.sig_Y1

        return self.silu_Y1 * self.Y2 
    
    def backward(self,grad_output):
        # dL/dY2 = grad_output * 门控值
        
        grad_Y2 = grad_output * self.silu_Y1

        # dL/d(SiLU_Y1) = grad_output * Y2
        grad_silu_Y1 = grad_output * self.Y2

        # d(SiLU_Y1)/dY1 的导数 (复用刚才推导的 SiLU 导数)
        d_silu_dY1 = self.sig_Y1 + self.Y1 * self.sig_Y1 * (1 - self.sig_Y1)

        # dL/dY1 = dL/d(SiLU_Y1) * d(SiLU_Y1)/dY1
        grad_Y1 = grad_silu_Y1 * d_silu_dY1

        return grad_Y1, grad_Y2



