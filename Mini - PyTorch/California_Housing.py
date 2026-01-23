import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入你的模块
from linear import Linear
from activations import ReLU
from loss import MSEloss # ⚠️ 注意这里用 MSE
from optimizer import Adam, SGD
from batchnorm import BatchNorm
from dropout import Dropout

# 复用 Sequential 类 (为了节省篇幅，实际运行时请确保定义了它，或者引用上面的)
class Sequential:
    def __init__(self, layers): self.layers = layers
    def forward(self, x): 
        for l in self.layers: x = l.forward(x)
        return x
    def backward(self, g): 
        for l in reversed(self.layers): g = l.backward(g)
    def get_params(self): return [p for l in self.layers if hasattr(l,'params') for v in l.params.values() for p in [v]]
    def get_grads(self): return [g for l in self.layers if hasattr(l,'params') for v in l.grads.values() for g in [v]]
    def train(self): 
        for l in self.layers: l.train()
    def eval(self): 
        for l in self.layers: l.eval()

# 专门针对回归的数据加载器 (不做 One-Hot)
class RegLoader:
    def __init__(self, X, y, batch_size=64):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.indices = np.arange(X.shape[0])
    def __iter__(self):
        np.random.shuffle(self.indices)
        self.curr = 0
        return self
    def __next__(self):
        if self.curr >= self.X.shape[0]: raise StopIteration
        end = min(self.curr + self.batch_size, self.X.shape[0])
        idx = self.indices[self.curr:end]
        self.curr = end
        return self.X[idx], self.y[idx]

def build_reg_net():
    # 输入8个特征，输出1个房价
    return Sequential([
        Linear(8, 64), BatchNorm(64), ReLU(), Dropout(0.1),
        Linear(64, 32), BatchNorm(32), ReLU(),
        Linear(32, 1) # 回归层，无激活
    ])

def run_experiment(name, opt_class, lr, x_train, y_train, x_test, y_test):
    print(f"🏠 开始训练: {name} (lr={lr})")
    np.random.seed(42)
    model = build_reg_net()
    criterion = MSEloss()
    
    if name == "SGD": optimizer = opt_class(model.get_params(), lr=lr, momentum=0.9)
    else: optimizer = opt_class(model.get_params(), lr=lr)

    loader = RegLoader(x_train, y_train, batch_size=64)
    loss_history = []

    for epoch in range(20): # 回归通常需要多跑几轮
        model.train()
        epoch_loss = 0
        count = 0
        for x_b, y_b in loader:
            pred = model.forward(x_b)
            loss = criterion.forward(pred, y_b)
            model.backward(criterion.backward())
            optimizer.step(model.get_grads())
            epoch_loss += loss
            count += 1
        
        avg_loss = epoch_loss / count
        loss_history.append(avg_loss)
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch} | Train MSE: {avg_loss:.4f}")
            
    # 最终测试集 MSE
    model.eval()
    pred_test = model.forward(x_test)
    final_mse = np.mean((pred_test - y_test)**2)
    print(f"   最终测试集 MSE: {final_mse:.4f}")
    return loss_history

if __name__ == "__main__":
    data = fetch_california_housing()
    X, y = data.data, data.target.reshape(-1, 1)
    
    # 归一化 (回归任务非常重要！)
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    loss_sgd = run_experiment("SGD", SGD, 0.001, x_train, y_train, x_test, y_test)
    loss_adam = run_experiment("Adam", Adam, 0.01, x_train, y_train, x_test, y_test)

    plt.plot(loss_sgd, label='SGD')
    plt.plot(loss_adam, label='Adam')
    plt.title('Housing Price Prediction MSE (Lower is Better)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()