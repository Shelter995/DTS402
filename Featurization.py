import pandas as pd

# 1. 读取数据
df = pd.read_csv('D:/Projects/DTS402/dataset/small_dataset_3k.csv')

# --- 预处理：确保 text 列是字符串，处理空值 ---
df['text'] = df['text'].fillna('').astype(str)

# --- 特征工程：一次性提取多个统计特征 ---

# 1. 基础长度特征
# 词数 (Word Count)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

# 2. 标点符号特征 (Punctuation)
# 虚假评论（特别是刷单的）有时会滥用感叹号
df['excl_count'] = df['text'].apply(lambda x: x.count('!'))

# --- 查看结果 ---
print("前5行数据特征预览：")
print(df[['label', 'text', 'word_count', 'excl_count']].head())

# --- 简单验证特征有效性 ---
# 我们可以看一下 CG (虚假) 和 OR (真实) 在这些特征上的平均值区别
print("\n不同类别的特征均值对比：")
print(df.groupby('label')[['word_count', 'excl_count']].mean())