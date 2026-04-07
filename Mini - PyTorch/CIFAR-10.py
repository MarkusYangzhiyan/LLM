import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


from linear import Linear
from activations import ReLU
from loss import CrossEntropyloss
from optimizer import Adam, SGD
from dropout import Dropout
from batchnorm import BatchNorm
from utils import calculate_accuracy
from dataloader import DataLoader

# 复用 Sequential 
class Sequential:
    def __init__(self, layers): self.layers = layers
    def forward(self, x): 
        for l in self.layers: x = l.forward(x)
        return x
    def backward(self, g): 
        for l in reversed(self.layers): g = l.backward(g)
    def train(self): 
        for l in self.layers: l.train()
    def eval(self): 
        for l in self.layers: l.eval()
    def get_params(self): return [p for l in self.layers if hasattr(l,'params') for v in l.params.values() for p in [v]]
    def get_grads(self): return [g for l in self.layers if hasattr(l,'params') for v in l.grads.values() for g in [v]]

def build_cifar_net():
    # 输入 3072 (32*32*3) -> 宽网络 1024 -> 256 -> 10
    return Sequential([
        Linear(3072, 1024), BatchNorm(1024), ReLU(), Dropout(0.2),
        Linear(1024, 256), BatchNorm(256), ReLU(), Dropout(0.2),
        Linear(256, 10)
    ])

def run_experiment(name, opt_class, lr, x_train, y_train, x_test, y_test):
    print(f" 开始训练: {name} (CIFAR-10 ...)")
    np.random.seed(42)
    model = build_cifar_net()
    criterion = CrossEntropyloss()
    
    if name == "SGD": optimizer = opt_class(model.get_params(), lr=lr, momentum=0.9)
    else: optimizer = opt_class(model.get_params(), lr=lr)

    # 加大 Batch Size 利用矩阵乘法加速
    train_loader = DataLoader(x_train, y_train, batch_size=256, shuffle=True)
    test_loader = DataLoader(x_test, y_test, batch_size=256, shuffle=False)
    
    history = []
    
    # 只跑3个Epoch演示效果
    for epoch in range(3): 
        model.train()
        total_loss = 0
        for i, (x_b, y_b) in enumerate(train_loader):
            logits = model.forward(x_b)
            loss = criterion.forward(logits, y_b)
            model.backward(criterion.backward())
            optimizer.step(model.get_grads())
            total_loss += loss
            
            # 打印进度条
            if i % 50 == 0: print(f"   Step {i} Loss: {loss:.4f}")

        # 测试
        model.eval()
        correct = 0
        total = 0
        for x_b, y_b in test_loader:
            out = model.forward(x_b)
            correct += calculate_accuracy(out, y_b) * x_b.shape[0]
            total += x_b.shape[0]
            
        acc = correct / total
        history.append(acc)
        print(f"   Epoch {epoch+1} Test Acc: {acc:.4f}")
        
    return history

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # 预处理：Flatten (N, 32, 32, 3) -> (N, 3072)
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # 对比
    acc_sgd = run_experiment("SGD", SGD, 0.01, x_train, y_train, x_test, y_test)
    acc_adam = run_experiment("Adam", Adam, 0.001, x_train, y_train, x_test, y_test)

    plt.plot(acc_sgd, label='SGD')
    plt.plot(acc_adam, label='Adam')
    plt.title('CIFAR-10 Accuracy (3 Epochs)')
    plt.legend()
    plt.show()
