# 第五步 (修正版): 先切分数据集，防止数据泄露
# The fifth step (Corrected): Split dataset FIRST to prevent data leakage.
import pandas as pd
import numpy as np
import os

# 1. 读取包含基础特征的文件 (这是第四步生成的)
input_path = r'D:\Projects\DTS402\dataset\merged_dataset_with_features.csv'
df = pd.read_csv(input_path)

# 简单的空值处理
df['text'] = df['text'].fillna('').astype(str)

print(f"原始数据集大小: {df.shape}")

# 2. 随机打乱并切分 (80% 训练, 20% 测试)
# 设置随机种子确保可复现
np.random.seed(42)

# 生成随机索引
indices = np.random.permutation(len(df))
test_size = int(len(df) * 0.2)

test_indices = indices[:test_size]
train_indices = indices[test_size:]

# 切分 DataFrame
df_train = df.iloc[train_indices].reset_index(drop=True)
df_test = df.iloc[test_indices].reset_index(drop=True)

print(f"训练集大小 (Train): {df_train.shape}")
print(f"测试集大小 (Test): {df_test.shape}")

# 3. 保存切分后的原始数据 (Raw Split Data)
# 这一步很重要，后续所有的特征工程都基于这两个分开的文件，互不干扰
train_save_path = r'D:\Projects\DTS402\dataset\train_raw.csv'
test_save_path = r'D:\Projects\DTS402\dataset\test_raw.csv'

df_train.to_csv(train_save_path, index=False)
df_test.to_csv(test_save_path, index=False)

print(f"\n✅ 数据集已切分并保存!")
print(f"训练集: {train_save_path}")
print(f"测试集: {test_save_path}")