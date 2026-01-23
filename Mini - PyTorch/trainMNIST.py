import numpy as np
import time
import matplotlib.pyplot as plt  # 引入画图库

# 1. 导入你的自定义模块
from layers import Layers
from linear import Linear
from activations import ReLU
from loss import CrossEntropyloss
from optimizer import Adam, SGD  # ⚠️ 记得确保 optimizer.py 里有 SGD 类
from dropout import Dropout
from batchnorm import BatchNorm
from utils import load_mnist_data, calculate_accuracy
from dataloader import DataLoader

# ==========================================
# 模型定义 (保持不变)
# ==========================================
class Model:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x):
        for layer in self.layers: x = layer.forward(x)
        return x
    def backward(self, grad):
        for layer in reversed(self.layers): grad = layer.backward(grad)
        return grad
    def train(self):
        for layer in self.layers: layer.train()
    def eval(self):
        for layer in self.layers: layer.eval()
    def get_params(self):
        return [p for layer in self.layers if hasattr(layer, 'params') for val in layer.params.values() for p in [val]]
    def get_grads(self):
        return [g for layer in self.layers if hasattr(layer, 'params') for val in layer.grads.values() for g in [val]]

class MNISTNet(Model):
    def __init__(self):
        # 结构: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Linear
        self.fc1 = Linear(784, 256, name="fc1")
        self.bn1 = BatchNorm(256, name="bn1")
        self.relu1 = ReLU(name="relu1")
        self.drop1 = Dropout(dropout=0.2, name="drop1")
        
        self.fc2 = Linear(256, 64, name="fc2")
        self.bn2 = BatchNorm(64, name="bn2")
        self.relu2 = ReLU(name="relu2")
        
        self.fc3 = Linear(64, 10, name="fc3_out")
        
        layers = [self.fc1, self.bn1, self.relu1, self.drop1, self.fc2, self.bn2, self.relu2, self.fc3]
        super().__init__(layers)

# ==========================================
# 核心：封装训练函数
# ==========================================
def run_experiment(optimizer_name, x_train, y_train, x_test, y_test, epochs=5):
    print(f"\n{'='*20} 开始训练: {optimizer_name} {'='*20}")
    
    # 1. ⚠️ 关键：每次实验都要重新初始化模型和种子，确保起点一致
    np.random.seed(42) 
    model = MNISTNet()
    
    # 2. 准备数据
    train_loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(x_test, y_test, batch_size=64, shuffle=False)
    criterion = CrossEntropyloss()
    
    # 3. 选择优化器
    # SGD 通常需要较大的学习率 (0.01 或 0.1) 才能动起来
    # Adam 通常需要较小的学习率 (0.001)
    if optimizer_name == "SGD":
        optimizer = SGD(model.get_params(), lr=0.01, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = Adam(model.get_params(), lr=0.001)
    
    # 记录历史数据用于画图
    history = {'loss': [], 'acc': []}
    
    # 4. 训练循环
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        for x_batch, y_batch_onehot in train_loader:
            # Forward
            logits = model.forward(x_batch)
            loss = criterion.forward(logits, y_batch_onehot)
            
            # Backward
            grad = criterion.backward()
            model.backward(grad)
            
            # Update
            optimizer.step(model.get_grads())
            
            total_loss += loss
            total_acc += calculate_accuracy(logits, y_batch_onehot)
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        # 记录
        history['loss'].append(avg_loss)
        
        # 测试集验证
        model.eval()
        test_acc_sum = 0
        test_cnt = 0
        for x_test_b, y_test_b in test_loader:
            out = model.forward(x_test_b)
            test_acc_sum += calculate_accuracy(out, y_test_b)
            test_cnt += 1
        avg_test_acc = test_acc_sum / test_cnt
        history['acc'].append(avg_test_acc)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | Test Acc: {avg_test_acc:.4f} | Time: {time.time()-start_time:.2f}s")
        
    return history

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 加载数据
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    if x_train is None:
        exit()

    # 1. 跑 SGD
    hist_sgd = run_experiment("SGD", x_train, y_train, x_test, y_test, epochs=5)

    # 2. 跑 Adam
    hist_adam = run_experiment("Adam", x_train, y_train, x_test, y_test, epochs=5)

    # ==========================================
    # 结果对比与画图
    # ==========================================
    print("\n" + "="*50)
    print("最终结果对比")
    print(f"SGD  最终 Loss: {hist_sgd['loss'][-1]:.4f} | 最终 Test Acc: {hist_sgd['acc'][-1]*100:.2f}%")
    print(f"Adam 最终 Loss: {hist_adam['loss'][-1]:.4f} | 最终 Test Acc: {hist_adam['acc'][-1]*100:.2f}%")
    print("="*50)

    # 画图
    epochs = range(1, 6)
    plt.figure(figsize=(12, 5))

    # 子图1: Loss 对比
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist_sgd['loss'], 'r-o', label='SGD (lr=0.01)')
    plt.plot(epochs, hist_adam['loss'], 'b-o', label='Adam (lr=0.001)')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 子图2: Accuracy 对比
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist_sgd['acc'], 'r-o', label='SGD Test Acc')
    plt.plot(epochs, hist_adam['acc'], 'b-o', label='Adam Test Acc')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    print("正在显示对比图...")
    plt.show()