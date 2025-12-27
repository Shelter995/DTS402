# 补充步骤：读取处理好的数据并准备 X_train, y_train
import pandas as pd
import numpy as np

# 1. 读取我们在 Cell 6 中生成的最终文件
train_path = r'D:\Projects\DTS402\dataset\train_processed_final.csv'
test_path = r'D:\Projects\DTS402\dataset\test_processed_final.csv'

df_train_final = pd.read_csv(train_path)
df_test_final = pd.read_csv(test_path)

print(f"数据加载成功 - 训练集: {df_train_final.shape}, 测试集: {df_test_final.shape}")

# 2. 提取特征和标签
# 排除非数值特征列 (category, text, label 以及可能存在的 dataset_source)
# 注意：我们保留 word_count, excl_count 和 tfidf_svd_xx 列
exclude_cols = ['category', 'text', 'label', 'dataset_source']
feature_cols = [c for c in df_train_final.columns if c not in exclude_cols]

print(f"使用的特征列 ({len(feature_cols)}列): {feature_cols[:5]} ...")

# 转换 X
X_train_raw = df_train_final[feature_cols].values.astype(float)
X_test_raw = df_test_final[feature_cols].values.astype(float)

# 转换 y (Label Encoding: OR -> 1, CG -> 0)
y_train = np.where(df_train_final['label'] == 'OR', 1, 0).reshape(-1, 1)
y_test = np.where(df_test_final['label'] == 'OR', 1, 0).reshape(-1, 1)

# 3. 标准化 (Normalization)
# 同样遵循"防止数据泄露"原则：仅在训练集上计算均值和方差
X_mean = np.mean(X_train_raw, axis=0)
X_std = np.std(X_train_raw, axis=0) + 1e-8 # 防止除以0

X_train = (X_train_raw - X_mean) / X_std
X_test = (X_test_raw - X_mean) / X_std

print("\n数据准备就绪!")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")