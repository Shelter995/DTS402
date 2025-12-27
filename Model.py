# The eighth step: Building the model and define the training function.

# ==========================================
# 3. Model Construction - MLP
# ==========================================
# 第七步：构建改进后的 MLP 模型 (解决过深和梯度问题)
import numpy as np
import pickle
import os

# 确保输出目录存在
output_dir = r'D:\Projects\DTS402\output\model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class SimpleMLP:
    def __init__(self, input_size, learning_rate=0.0003, momentum=0.9, reg_lambda=0.01):
        """
        改进点 1: 降低学习率 (0.001 -> 0.0003) 以防止震荡
        改进点 2: 增加正则化权重 reg_lambda 以防止过拟合
        """
        self.lr = learning_rate
        self.momentum = momentum
        self.reg_lambda = reg_lambda

        # 改进点 3: 简化网络结构 (5层变3层)
        # 结构: Input -> 64 (Hidden 1) -> 32 (Hidden 2) -> 1 (Output)
        # 这种"金字塔"结构对于几千条数据的分类任务更稳健

        np.random.seed(42)

        # Layer 1: Input -> 64
        self.W1 = np.random.randn(input_size, 64) * np.sqrt(2.0 / input_size)  # He Init
        self.b1 = np.zeros((1, 64))

        # Layer 2: 64 -> 32
        self.W2 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros((1, 32))

        # Layer 3: 32 -> 1 (Output)
        self.W3 = np.random.randn(32, 1) * np.sqrt(1.0 / 32)  # Xavier Init for Sigmoid
        self.b3 = np.zeros((1, 1))

        # Momentum (动量) 缓存初始化
        self.vW1, self.vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vW2, self.vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.vW3, self.vb3 = np.zeros_like(self.W3), np.zeros_like(self.b3)

        # Batch Normalization 参数
        self.gamma1, self.beta1 = np.ones((1, 64)), np.zeros((1, 64))
        self.gamma2, self.beta2 = np.ones((1, 32)), np.zeros((1, 32))

        # BN 运行时统计量
        self.running_mean1, self.running_var1 = np.zeros((1, 64)), np.ones((1, 64))
        self.running_mean2, self.running_var2 = np.zeros((1, 32)), np.ones((1, 32))

        # 用于存储最佳模型
        self.best_weights = None
        self.best_f1 = -1

        # 改进点 4: 缓存 Dropout Mask，防止反向传播梯度错误
        self.mask1 = None
        self.mask2 = None

    def batch_norm(self, x, gamma, beta, running_mean, running_var, training=True, momentum=0.9):
        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)

            # 更新全局统计量
            running_mean[:] = momentum * running_mean + (1 - momentum) * mean
            running_var[:] = momentum * running_var + (1 - momentum) * var

            x_norm = (x - mean) / np.sqrt(var + 1e-8)
        else:
            x_norm = (x - running_mean) / np.sqrt(running_var + 1e-8)

        return gamma * x_norm + beta

    def dropout(self, x, drop_rate=0.3, training=True):
        if training:
            # Inverted Dropout
            mask = np.random.binomial(1, 1 - drop_rate, size=x.shape) / (1 - drop_rate)
            return x * mask, mask  # 返回 mask 以便反向传播使用
        return x, None

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def forward(self, X, training=True):
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.z1_bn = self.batch_norm(self.z1, self.gamma1, self.beta1,
                                     self.running_mean1, self.running_var1, training)
        self.a1 = self.relu(self.z1_bn)
        # 保存 mask1
        self.a1_drop, self.mask1 = self.dropout(self.a1, 0.4, training)  # 稍微增加一点 Dropout (0.3 -> 0.4)

        # Layer 2
        self.z2 = np.dot(self.a1_drop, self.W2) + self.b2
        self.z2_bn = self.batch_norm(self.z2, self.gamma2, self.beta2,
                                     self.running_mean2, self.running_var2, training)
        self.a2 = self.relu(self.z2_bn)
        # 保存 mask2
        self.a2_drop, self.mask2 = self.dropout(self.a2, 0.3, training)

        # Layer 3 (Output)
        self.z3 = np.dot(self.a2_drop, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)

        return self.a3

    def backward_and_update(self, X, y, output):
        m = X.shape[0]

        # Output Layer Gradients
        dz3 = output - y
        dW3 = (1 / m) * np.dot(self.a2_drop.T, dz3) + (self.reg_lambda / m) * self.W3
        db3 = (1 / m) * np.sum(dz3, axis=0, keepdims=True)

        # Layer 2 Gradients
        da2 = np.dot(dz3, self.W3.T)
        # 关键修正: 应用前向传播时保存的 mask
        if self.mask2 is not None:
            da2 *= self.mask2

        dz2 = da2 * self.relu_derivative(self.z2_bn)
        dW2 = (1 / m) * np.dot(self.a1_drop.T, dz2) + (self.reg_lambda / m) * self.W2
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # Layer 1 Gradients
        da1 = np.dot(dz2, self.W2.T)
        # 关键修正: 应用 mask
        if self.mask1 is not None:
            da1 *= self.mask1

        dz1 = da1 * self.relu_derivative(self.z1_bn)
        dW1 = (1 / m) * np.dot(X.T, dz1) + (self.reg_lambda / m) * self.W1
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # SGD with Momentum Update
        # Layer 3
        self.vW3 = self.momentum * self.vW3 - self.lr * dW3
        self.vb3 = self.momentum * self.vb3 - self.lr * db3
        self.W3 += self.vW3
        self.b3 += self.vb3

        # Layer 2
        self.vW2 = self.momentum * self.vW2 - self.lr * dW2
        self.vb2 = self.momentum * self.vb2 - self.lr * db2
        self.W2 += self.vW2
        self.b2 += self.vb2

        # Layer 1
        self.vW1 = self.momentum * self.vW1 - self.lr * dW1
        self.vb1 = self.momentum * self.vb1 - self.lr * db1
        self.W1 += self.vW1
        self.b1 += self.vb1

        # (简化版：这里略去了 Gamma 和 Beta 的梯度更新，对于简单任务通常影响不大，保持代码简洁)

    def calculate_f1(self, X, y):
        preds = (self.forward(X, training=False) > 0.5).astype(int)
        tp = np.sum((y == 1) & (preds == 1))
        # fp = np.sum((y == 0) & (preds == 1))
        # fn = np.sum((y == 1) & (preds == 0))
        prec = tp / (np.sum(preds == 1) + 1e-8)
        rec = tp / (np.sum(y == 1) + 1e-8)
        return 2 * (prec * rec) / (prec + rec + 1e-8)

    def calculate_precision(self, X, y):
        preds = (self.forward(X, training=False) > 0.5).astype(int)
        tp = np.sum((y == 1) & (preds == 1))
        fp = np.sum((y == 0) & (preds == 1))
        return tp / (tp + fp + 1e-8)

    def train(self, X_train, y_train, X_val, y_val, epochs=2000, batch_size=64):
        history = {'loss': [], 'train_precision': [], 'val_precision': []}
        n_samples = X_train.shape[0]

        # Early Stopping 设置
        patience = 300  # 增加 patience，因为学习率降低了，收敛会变慢
        best_val_f1 = -1
        patience_counter = 0

        print(f"开始训练 (Epochs={epochs}, LR={self.lr})...")
        print(f"网络结构: {X_train.shape[1]} -> 64 -> 32 -> 1")

        for epoch in range(epochs):
            # 学习率衰减
            if epoch > 0 and epoch % 200 == 0:
                self.lr *= 0.8
                print(f" -> Epoch {epoch}: 学习率衰减为 {self.lr:.8f}")

            # Shuffle
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            # Mini-batch Training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.forward(X_batch, training=True)
                self.backward_and_update(X_batch, y_batch, output)

                # Binary Cross Entropy Loss
                batch_loss = -np.mean(y_batch * np.log(output + 1e-8) +
                                      (1 - y_batch) * np.log(1 - output + 1e-8))
                epoch_loss += batch_loss * X_batch.shape[0]

            avg_loss = epoch_loss / n_samples
            history['loss'].append(avg_loss)

            # 评估
            if epoch % 10 == 0:
                train_prec = self.calculate_precision(X_train, y_train)
                val_prec = self.calculate_precision(X_val, y_val)
                val_f1 = self.calculate_f1(X_val, y_val)

                history['train_precision'].append(train_prec)
                history['val_precision'].append(val_prec)

                # 保存最佳模型
                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.best_weights = {
                        'W1': self.W1.copy(), 'b1': self.b1.copy(),
                        'W2': self.W2.copy(), 'b2': self.b2.copy(),
                        'W3': self.W3.copy(), 'b3': self.b3.copy(),
                        'gamma1': self.gamma1.copy(), 'beta1': self.beta1.copy(),
                        'gamma2': self.gamma2.copy(), 'beta2': self.beta2.copy(),
                        'running_mean1': self.running_mean1.copy(),
                        'running_var1': self.running_var1.copy(),
                        'running_mean2': self.running_mean2.copy(),
                        'running_var2': self.running_var2.copy()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 10

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Train Prec={train_prec:.4f}, Val F1={val_f1:.4f}")

                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"Best validation F1: {self.best_f1:.4f}")
                    break

        # 保存最终模型
        self.save_model(r"D:\Projects\DTS402\output\model\model_final.pkl")
        if self.best_weights:
            self._load_best_weights()
            self.save_model(r"D:\Projects\DTS402\output\model\model_best.pkl")

        return history

    def _load_best_weights(self):
        if self.best_weights:
            self.W1 = self.best_weights['W1']
            self.b1 = self.best_weights['b1']
            self.W2 = self.best_weights['W2']
            self.b2 = self.best_weights['b2']
            self.W3 = self.best_weights['W3']
            self.b3 = self.best_weights['b3']
            self.gamma1 = self.best_weights['gamma1']
            self.beta1 = self.best_weights['beta1']
            self.gamma2 = self.best_weights['gamma2']
            self.beta2 = self.best_weights['beta2']
            self.running_mean1 = self.best_weights['running_mean1']
            self.running_var1 = self.best_weights['running_var1']
            self.running_mean2 = self.best_weights['running_mean2']
            self.running_var2 = self.best_weights['running_var2']

    def save_model(self, filename):
        model_params = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'gamma1': self.gamma1, 'beta1': self.beta1,
            'gamma2': self.gamma2, 'beta2': self.beta2,
            'running_mean1': self.running_mean1,
            'running_var1': self.running_var1,
            'running_mean2': self.running_mean2,
            'running_var2': self.running_var2
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
        print(f"模型已保存: {filename}")

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        self.W1 = params['W1']
        self.b1 = params['b1']
        self.W2 = params['W2']
        self.b2 = params['b2']
        self.W3 = params['W3']
        self.b3 = params['b3']
        self.gamma1 = params['gamma1']
        self.beta1 = params['beta1']
        self.gamma2 = params['gamma2']
        self.beta2 = params['beta2']
        self.running_mean1 = params['running_mean1']
        self.running_var1 = params['running_var1']
        self.running_mean2 = params['running_mean2']
        self.running_var2 = params['running_var2']

    def predict(self, X):
        output = self.forward(X, training=False)
        return (output > 0.5).astype(int)


print("SimpleMLP 类定义完成 (3层网络结构，低学习率，修复Dropout)")