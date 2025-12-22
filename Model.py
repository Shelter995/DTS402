# ==========================================
# 3. 构建模型 (Model Construction - MLP)
# ==========================================
import numpy as np


class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate

        # 初始化权重和偏置 (He Initialization or Random)
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # clip防止溢出

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)  # 隐藏层激活

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # 输出层激活 (预测值)
        return self.a2

    def backward(self, X, y, output):
        # 反向传播 (Gradient Descent)
        m = X.shape[0]

        # 计算输出层误差: Loss 对 z2 的导数 = (prediction - y)
        dz2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # 计算隐藏层误差
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # 更新权重
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=1000):
        loss_history = []
        for epoch in range(epochs):
            # 1. Forward
            output = self.forward(X)

            # 2. Compute Loss (Binary Cross Entropy)
            # 加 1e-8 防止 log(0)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            loss_history.append(loss)

            # 3. Backward & Update
            self.backward(X, y, output)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return loss_history

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)