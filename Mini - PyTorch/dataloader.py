import numpy as np
from utils import to_one_hot # 导入刚才写的工具

class DataLoader:
    def __init__(self, X, y, batch_size=64, shuffle=True):
        """
        X: 输入数据 (N, 784)
        y: 标签数据 (N,)
        batch_size: 每次喂给模型多少个样本
        shuffle: 是否打乱顺序 (训练时必须 True，测试时 False)
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        # 计算总共有多少个 batch (向上取整)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
        # 初始化索引 [0, 1, 2, ... N-1]
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        """
        当使用 for x, y in loader 时，会触发这个函数
        """
        # 每次重新开始遍历时，如果是训练模式，就打乱索引
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current_idx = 0
        return self

    def __next__(self):
        """
        每次循环获取下一个 batch
        """
        # 如果游标已经到了最后，停止迭代
        if self.current_idx >= self.num_samples:
            raise StopIteration
        
        # 确定当前 Batch 的起止位置
        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        
        # 获取这一个 Batch 对应的乱序索引
        batch_indices = self.indices[self.current_idx : end_idx]
        
        # 1. 取出数据
        X_batch = self.X[batch_indices] # (Batch, 784)
        y_batch = self.y[batch_indices] # (Batch, )
        
        # 2. 关键步骤：把标签转成 One-Hot
        # 因为你的 CrossEntropyLoss 需要 shape 为 (Batch, 10) 的 y
        y_batch_onehot = to_one_hot(y_batch, num_classes=10)
        
        # 更新游标
        self.current_idx = end_idx
        
        return X_batch, y_batch_onehot

    def __len__(self):
        # 允许使用 len(loader) 查看有多少个 batch
        return self.num_batches