import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 设置绘图风格
# ==========================================
sns.set(style="whitegrid", context="talk")
# Windows系统若中文乱码，请解开下面这行的注释
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取数据 (请修改为您真实的文件名)
# ==========================================
# 假设文件名如下，请按需修改
file_open = r'D:\Projects\DTS402\dataset\fake reviews dataset.csv'
file_self = r'D:\Projects\DTS402\dataset\self_dataset.csv'

try:
    df_open = pd.read_csv(file_open)
    df_self = pd.read_csv(file_self)
    print("✅ 数据读取成功")
except FileNotFoundError:
    print("❌ 错误：找不到文件，请确认CSV文件在当前目录下。")
    # 为了演示代码运行，这里生成少量模拟数据，您运行时可忽略这部分
    df_open = pd.DataFrame({'text_': ['good']*500, 'rating': [4]*500, 'label': ['CG']*250+['OR']*250})
    df_self = pd.DataFrame({'text': ['bad']*200, 'rating': [1]*200, 'label': [1]*100+[0]*100})

# ==========================================
# 2. 数据预处理与对齐 (关键步骤)
# ==========================================

# # 2.1 统一文本列名
# # 开源数据通常叫 'text_'，自采叫 'text'
# if 'text_' in df_open.columns:
#     df_open.rename(columns={'text_': 'text'}, inplace=True)

# 2.2 添加来源标签 (Source Tagging)
df_open['source'] = 'Open Source'
df_self['source'] = 'Self-collected'

# 2.3 统一真假标签 (Label Standardization)
# 假设：开源用 'CG'/'OR'，自采用 1/0 (或 'Fake'/'Real')
# 我们统一映射为：'Fake' 和 'Real'
def normalize_label(val):
    if val in ['CG', 1, '1', 'Fake']:
        return 'Fake'
    elif val in ['OR', 0, '0', 'Real']:
        return 'Real'
    return 'Unknown'

df_open['label_norm'] = df_open['label'].apply(normalize_label)
df_self['label_norm'] = df_self['label'].apply(normalize_label)

# 2.4 计算文本长度 (Text Length)
df_open['text_length'] = df_open['text'].apply(lambda x: len(str(x).split()))
df_self['text_length'] = df_self['text'].apply(lambda x: len(str(x).split()))

# ==========================================
# 3. 合并数据集
# ==========================================
cols = ['source', 'label_norm', 'rating', 'text_length']
df_all = pd.concat([df_open[cols], df_self[cols]], ignore_index=True)

# ==========================================
# 4. 生成统计摘要 (均值、方差)
# ==========================================
print("\n=== 统计摘要 (均值 Mean & 标准差 Std) ===")
# std (标准差) 的平方就是方差
summary = df_all.groupby(['source', 'label_norm'])[['rating', 'text_length']].agg(['mean', 'std'])
print(summary)

# ==========================================
# 5. 可视化对比
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 图1: 评分分布 (修正：使用百分比而非绝对数量)
sns.histplot(
    data=df_all,
    x='rating',
    hue='source',
    multiple='dodge',
    stat='percent',      # <--- 关键修改：显示百分比
    common_norm=False,   # <--- 关键修改：分别计算百分比(蓝色总和100%，绿色总和100%)
    bins=5,
    shrink=0.8,
    palette='viridis',
    ax=axes[0]
)
axes[0].set_title('Rating Distribution (Normalized %)')
axes[0].set_xlabel('Rating (1-5)')
axes[0].set_ylabel('Percentage (%)') # Y轴标签改为百分比

# 图2: 文本长度箱线图 (保持不变)
sns.boxplot(
    data=df_all,
    x='source',
    y='text_length',
    hue='label_norm',
    palette='Set2',
    showfliers=False,
    ax=axes[1]
)
axes[1].set_title('Text Length Distribution')
axes[1].set_xlabel('Dataset Source')
axes[1].set_ylabel('Text Word Count')

plt.tight_layout()
plt.show()