import pandas as pd

# 1. 读取数据
input_path = r'C:\Users\Zijie.Xue25\XZJ_PythonProjects\DTS402\dataset\open_3k.csv'
df = pd.read_csv(input_path)

# --- 预处理：确保 text 列是字符串，处理空值 ---
df['text'] = df['text'].fillna('').astype(str)

# --- 特征工程：计算特征并添加为新列 ---

# 1. 基础长度特征
# 词数 (Word Count)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

# 2. 标点符号特征 (Punctuation)
df['excl_count'] = df['text'].apply(lambda x: x.count('!'))

# --- 查看结果 (可选，用于确认) ---
print("前5行数据特征预览：")
print(df[['label', 'text', 'word_count', 'excl_count']].head())
# --- 简单验证特征有效性 ---
# 我们可以看一下 CG (虚假) 和 OR (真实) 在这些特征上的平均值区别
print("\n不同类别的特征均值对比：")
print(df.groupby('label')[['word_count', 'excl_count']].mean())

# --- 【关键修改】将结果保存到 CSV 文件 ---
# 建议保存为新文件名，以区分原始数据
output_path = r'C:\Users\Zijie.Xue25\XZJ_PythonProjects\DTS402\dataset\open_3k_with_features.csv'

# index=False 代表不保存行索引(0,1,2...)，只保存数据列
df.to_csv(output_path, index=False)

print(f"\n成功！新特征已保存到文件：{output_path}")