# ==========================================
# 3. 构建模型 (Model Construction - MLP)
# ==========================================
import numpy as np
import pickle # 用于保存模型参数 (Python标准库，符合规定)
import copy   # 用于复制最佳模型参数
import os


class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, momentum=0.9, reg_lambda=0.01):
        self.lr = learning_rate
        self.momentum = momentum  # 新增动量因子 (通常设为 0.9)
        self.reg_lambda = reg_lambda  # L2 正则化系数

        # 初始化权重和偏置 (He Initialization or Random)
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # 初始化动量速度 (Velocity)，用于记录梯度的“惯性”
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

        # 用于记录最佳参数
        self.best_weights = None
        self.best_f1 = -1  # 初始化最佳 F1 分数

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # clip防止溢出

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    # 【新增】ReLU 激活函数
    def relu(self, z):
        return np.maximum(0, z)

    # 【新增】ReLU 的导数
    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # 隐藏层激活

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # 输出层激活 (预测值)
        return self.a2

    def backward_and_update(self, X, y, output):
        # 反向传播 (Gradient Descent)
        m = X.shape[0]

        # 计算输出层误差: Loss 对 z2 的导数 = (prediction - y)
        dz2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # 计算隐藏层误差
        # 误差反向传递到第一层
        da1 = np.dot(dz2, self.W2.T)
        # 乘以 ReLU 的导数
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # 加入 L2 正则化梯度项
        # dW += (lambda / m) * W
        dW2 += (self.reg_lambda / m) * self.W2
        dW1 += (self.reg_lambda / m) * self.W1

        # 更新权重
        # v = momentum * v - learning_rate * gradient
        # W = W + v
        self.vW2 = self.momentum * self.vW2 - self.lr * dW2
        self.vb2 = self.momentum * self.vb2 - self.lr * db2
        self.vW1 = self.momentum * self.vW1 - self.lr * dW1
        self.vb1 = self.momentum * self.vb1 - self.lr * db1

        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1
        # self.W1 -= self.lr * dW1
        # self.b1 -= self.lr * db1
        # self.W2 -= self.lr * dW2
        # self.b2 -= self.lr * db2

    # 计算精确率的辅助函数
    def calculate_precision(self, X, y):
        preds = (self.forward(X) > 0.5).astype(int)
        y = y.reshape(-1, 1)
        tp = np.sum((y == 1) & (preds == 1))
        fp = np.sum((y == 0) & (preds == 1))
        return tp / (tp + fp + 1e-8)  # 防止除以0

    def calculate_f1(self, X, y):
        # 简单计算 F1 用于选取最佳模型
        preds = (self.forward(X) > 0.5).astype(int)
        tp = np.sum((y == 1) & (preds == 1))
        fp = np.sum((y == 0) & (preds == 1))
        fn = np.sum((y == 1) & (preds == 0))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        return 2 * (prec * rec) / (prec + rec + 1e-8)

    def save_model(self, filename):
        """保存当前权重到文件"""
        model_params = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
        print(f"模型已保存: {filename}")

    def load_model(self, filename):
        """加载权重"""
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        self.W1 = params['W1'];
        self.b1 = params['b1']
        self.W2 = params['W2'];
        self.b2 = params['b2']

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, batch_size=64):
        # 历史记录容器
        history = {
            'loss': [],
            'train_precision': [],
            'val_precision': []
        }
        n_samples = X_train.shape[0]

        print(f"开始训练 ({epochs} Epochs)...")

        for epoch in range(epochs):
            # 【新增】学习率衰减：每 500 轮，学习率乘 0.5
            if epoch > 0 and epoch % 500 == 0:
                self.lr *= 0.5
                print(f" -> Learning rate decayed to {self.lr:.6f}")

            # 1. 每个 Epoch 开始前打乱数据 (Shuffle)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            # 2. Mini-Batch 迭代
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i: i + batch_size]
                y_batch = y_shuffled[i: i + batch_size]

                # 前向与反向传播 (只针对当前 Batch)
                output = self.forward(X_batch)
                self.backward_and_update(X_batch, y_batch, output)

                # 累加 Loss (用于记录)
                batch_loss = -np.mean(y_batch * np.log(output + 1e-8) + (1 - y_batch) * np.log(1 - output + 1e-8))
                epoch_loss += batch_loss * X_batch.shape[0]

            # 计算平均 Loss
            avg_loss = epoch_loss / n_samples
            history['loss'].append(avg_loss)

            # 3. 记录指标 & 保存最佳模型
            if epoch % 10 == 0:  # 每10轮检查一次，减少计算量
                train_prec = self.calculate_precision(X_train, y_train)
                val_prec = self.calculate_precision(X_val, y_val)
                val_f1 = self.calculate_f1(X_val, y_val)

                history['train_precision'].append(train_prec)
                history['val_precision'].append(val_prec)

                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.best_weights = {'W1': self.W1.copy(), 'b1': self.b1.copy(), 'W2': self.W2.copy(),
                                         'b2': self.b2.copy()}

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val F1={val_f1:.4f}")

        # 保存
        self.save_model("D:/Projects/DTS402/output/model/model_final.pkl")
        if self.best_weights:
            # 恢复最佳权重保存
            temp_W1, temp_b1 = self.W1, self.b1
            temp_W2, temp_b2 = self.W2, self.b2

            self.W1, self.b1 = self.best_weights['W1'], self.best_weights['b1']
            self.W2, self.b2 = self.best_weights['W2'], self.best_weights['b2']
            self.save_model("D:/Projects/DTS402/output/model/model_best.pkl")

            # 恢复回当前权重，或者就保留最佳权重
            self.W1, self.b1 = temp_W1, temp_b1
            self.W2, self.b2 = temp_W2, temp_b2

        return history

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)