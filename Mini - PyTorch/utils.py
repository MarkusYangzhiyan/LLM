import numpy as np

def to_one_hot(y, num_classes=10):
    """
    将整数标签转为 One-Hot 编码
    输入 y: [2, 0, 1]                             
    true label   with batch_size = 3 as an example 

    输出: [[0, 0, 1], [1, 0, 0], [0, 1, 0]]  

    """

    # 如果已经是 One-Hot 的话就不处理，直接返回本身
    if y.ndim > 1: return y  
    
    # 获取样本数量   y = [2,0,1] -->  y.shape[0] = 3 
    N = y.shape[0]


    one_hot = np.zeros((N, num_classes))
  
    one_hot[np.arange(N), y] = 1
    return one_hot

def calculate_accuracy(probs, y_true):
    """
    计算准确率
    probs: 模型输出的概率矩阵 (N, 10)
    y_true: 真实的标签，支持One-Hot或整数索引
    """
    # 找到概率最大的那个下标 (预测出的数字)
    y_pred = np.argmax(probs, axis=1)
    
    # 如果 y_true 是 One-Hot (二维)，先把它转回整数 (一维)
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
        
    # 计算预测正确的比例
    accuracy = np.mean(y_pred == y_true)
    return accuracy

def load_mnist_data():
    """
    用 keras/tensorflow 加载数据。
    """
    
    from tensorflow.keras.datasets import mnist
       
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
    # 预处理 1: Flatten
    # (60000, 28, 28) -> (60000, 784)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
        
    # 预处理 2: 归一化 
    # 把像素从 0-255 变成 0-1 的浮点数
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
        
    return x_train, y_train, x_test, y_test
        
