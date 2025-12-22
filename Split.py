import numpy as np
import pandas as pd

from Data_Embedding_demo import final_df

# 假设 final_df 是我们上一通过 SVD 得到的 DataFrame
# final_df 包含: category, rating, label, word_count, excl_count, tfidf_svd_0...tfidf_svd_23

# ==========================================
# 1. 数据准备 (Data Preparation)
# ==========================================

# 1.1 提取特征 (X) 和 标签 (y)
# 除去非数值列 (category, text) 和 标签列 (label)
feature_cols = [c for c in final_df.columns if c not in ['category', 'text', 'label']]
X = final_df[feature_cols].values.astype(float)

# 1.2 标签编码 (Label Encoding)
# 假设 'OR' (Original?) 是 1, 'CG' (Computer Generated?) 是 0
# 您需要根据实际业务含义调整
y = np.where(final_df['label'] == 'OR', 1, 0)
y = y.reshape(-1, 1)  # 变成 (N, 1) 的形状

# 1.3 数据归一化 (Normalization) - 神经网络对尺度非常敏感！
# 手写 StandardScaler: (X - mean) / std
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0) + 1e-8  # 加一个小数值防止除以0
X_normalized = (X - X_mean) / X_std


# ==========================================
# 2. 划分数据集 (Train/Test Split)
# ==========================================
# 严禁使用 sklearn.model_selection.train_test_split

def manual_train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)  # 随机打乱索引

    test_samples = int(n_samples * test_size)

    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


X_train, X_test, y_train, y_test = manual_train_test_split(X_normalized, y, test_size=0.2)

# print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")