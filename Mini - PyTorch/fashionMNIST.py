import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

from linear import Linear
from activations import ReLU
from loss import CrossEntropyloss
from optimizer import Adam, SGD
from dropout import Dropout
from batchnorm import BatchNorm
from utils import calculate_accuracy
from dataloader import DataLoader

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x):
        for layer in self.layers: x = layer.forward(x)
        return x
    def backward(self, grad):
        for layer in reversed(self.layers): grad = layer.backward(grad)
    def train(self):
        for layer in self.layers: layer.train()
    def eval(self):
        for layer in self.layers: layer.eval()
    def get_params(self):
        return [p for layer in self.layers if hasattr(layer, 'params') for v in layer.params.values() for p in [v]]
    def get_grads(self):
        return [g for layer in self.layers if hasattr(layer, 'params') for v in layer.grads.values() for g in [v]]

# 定义网络
def build_fashion_net():
    # 结构: 784 -> 256 -> 128 -> 10
    layers = [
        Linear(784, 256), BatchNorm(256), ReLU(), Dropout(0.2),
        Linear(256, 128), BatchNorm(128), ReLU(),
        Linear(128, 10)
    ]
    return Sequential(layers)

# 实验逻辑
def run_experiment(name, opt_class, lr, x_train, y_train, x_test, y_test):
    print(f"开始训练: {name} (lr={lr})")
    np.random.seed(42)
    model = build_fashion_net()
    criterion = CrossEntropyloss()
    
    # 区分SGD需要momentum参数
    if name == "SGD":
        optimizer = opt_class(model.get_params(), lr=lr, momentum=0.9)
    else:
        optimizer = opt_class(model.get_params(), lr=lr)

    train_loader = DataLoader(x_train, y_train, batch_size=128, shuffle=True)
    test_loader = DataLoader(x_test, y_test, batch_size=128, shuffle=False)
    
    loss_history = []
    acc_history = []
    
    for epoch in range(5): 
        model.train()
        epoch_loss = 0
        for x_batch, y_batch_onehot in train_loader:
            logits = model.forward(x_batch)
            loss = criterion.forward(logits, y_batch_onehot)
            model.backward(criterion.backward())
            optimizer.step(model.get_grads())
            epoch_loss += loss
            
        # 验证
        model.eval()
        correct = 0
        total = 0
        for x_b, y_b_onehot in test_loader:
            out = model.forward(x_b)
            correct += calculate_accuracy(out, y_b_onehot) * x_b.shape[0]
            total += x_b.shape[0]
            
        avg_loss = epoch_loss / len(train_loader)
        acc = correct / total
        loss_history.append(avg_loss)
        acc_history.append(acc)
        print(f"   Epoch {epoch+1} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")
        
    return loss_history, acc_history

if __name__ == "__main__":
   
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
   
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    # 对比
    loss_sgd, acc_sgd = run_experiment("SGD", SGD, 0.01, x_train, y_train, x_test, y_test)
    loss_adam, acc_adam = run_experiment("Adam", Adam, 0.001, x_train, y_train, x_test, y_test)

    # 画图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_sgd, label='SGD')
    plt.plot(loss_adam, label='Adam')
    plt.title('Fashion-MNIST Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(acc_sgd, label='SGD')
    plt.plot(acc_adam, label='Adam')
    plt.title('Fashion-MNIST Accuracy')
    plt.legend()
    plt.show()
