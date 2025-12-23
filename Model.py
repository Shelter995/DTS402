# ==========================================
# 3. 构建模型 (Model Construction - MLP)
# ==========================================
import numpy as np
import pickle # 用于保存模型参数 (Python标准库，符合规定)
import copy   # 用于复制最佳模型参数
import os


class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate

        # 初始化权重和偏置 (He Initialization or Random)
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # 用于记录最佳参数
        self.best_weights = None
        self.best_f1 = -1  # 初始化最佳 F1 分数

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
        print(f" -> 模型已保存: {filename}")

    def load_model(self, filename):
        """加载权重"""
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        self.W1 = params['W1'];
        self.b1 = params['b1']
        self.W2 = params['W2'];
        self.b2 = params['b2']

    def train(self, X_train, y_train, X_val, y_val, epochs=1000):
        # 历史记录容器
        history = {
            'loss': [],
            'train_precision': [],
            'val_precision': []
        }

        print(f"开始训练 ({epochs} Epochs)...")

        for epoch in range(epochs):
            # 1. Forward & Backward (Train)
            output = self.forward(X_train)
            self.backward(X_train, y_train, output)

            # 2. 记录 Loss
            loss = -np.mean(y_train * np.log(output + 1e-8) + (1 - y_train) * np.log(1 - output + 1e-8))
            history['loss'].append(loss)

            # 3. 计算并记录 Precision (Train & Val)
            train_prec = self.calculate_precision(X_train, y_train)
            val_prec = self.calculate_precision(X_val, y_val)

            history['train_precision'].append(train_prec)
            history['val_precision'].append(val_prec)

            # 4. 最佳模型检查 (基于验证集 F1 Score)
            # 我们用 F1 来判断哪个模型最好，因为它综合了 Precision 和 Recall
            current_val_f1 = self.calculate_f1(X_val, y_val)
            if current_val_f1 > self.best_f1:
                self.best_f1 = current_val_f1
                # 深度复制当前权重作为最佳权重
                self.best_weights = {
                    'W1': self.W1.copy(), 'b1': self.b1.copy(),
                    'W2': self.W2.copy(), 'b2': self.b2.copy()
                }

            if epoch % 200 == 0:
                print(f"    Epoch {epoch}: Loss={loss:.4f}, Val Precision={val_prec:.4f}, Best F1={self.best_f1:.4f}")

        # 训练结束后，保存两个模型

        # 1. 保存最后一轮模型 (Final Model)
        self.save_model("D:/Projects/DTS402/output/model/model_final.pkl")

        # 2. 保存并加载最佳模型 (Best Model)
        if self.best_weights:
            # 先暂存当前的 final 权重
            final_weights = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

            # 加载最佳权重并保存文件
            self.W1 = self.best_weights['W1'];
            self.b1 = self.best_weights['b1']
            self.W2 = self.best_weights['W2'];
            self.b2 = self.best_weights['b2']
            self.save_model("D:/Projects/DTS402/output/model/model_best.pkl")

            # 恢复回最后一轮的权重(以便画图反映的是完整的训练过程)
            # 或者您可以选择就让模型停留在最佳状态，这里我们选择恢复，让用户决定
            self.W1 = final_weights['W1'];
            self.b1 = final_weights['b1']
            self.W2 = final_weights['W2'];
            self.b2 = final_weights['b2']

        return history

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)