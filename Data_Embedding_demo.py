import pandas as pd
import numpy as np
import re
from collections import Counter
import random # 需要引入 random 库

# # --- 修正后的模拟数据生成 (确保词汇量 > 24) ---
# # print("正在生成更丰富的模拟数据...")
#
# # 1. 创建一个包含 100 个不同单词的“词库”
# vocab_pool = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew",
#     "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince", "raspberry",
#     "strawberry", "tangerine", "ugli", "vanilla", "watermelon", "xylocarp", "yam", "zucchini",
#     "book", "read", "story", "novel", "author", "page", "cover", "text",
#     "electronic", "device", "screen", "battery", "charge", "power", "button",
#     "home", "kitchen", "bed", "bath", "clean", "cook", "wash", "sleep",
#     "good", "bad", "great", "terrible", "love", "hate", "best", "worst",
#     "amazing", "awful", "nice", "poor", "excellent", "disappointed",
#     "happy", "sad", "angry", "joy", "fear", "surprise", "trust",
#     "fast", "slow", "shipping", "delivery", "box", "package", "arrive"]
#
# # 2. 随机生成 4000 条评论，每条评论包含 5-15 个随机词
# dummy_texts = []
# for _ in range(4000):
#     # 随机选取 5 到 15 个词组成一句话
#     sentence = " ".join(random.choices(vocab_pool, k=random.randint(5, 15)))
#     dummy_texts.append(sentence)
#
# df = pd.DataFrame({
#     'category': np.random.choice(['Electronics', 'Books', 'Home'], 4000),
#     'rating': np.random.randint(1, 6, 4000),
#     'label': np.random.choice(['CG', 'OR'], 4000),
#     'text': dummy_texts,  # 使用新的随机文本
#     'word_count': np.random.randint(5, 50, 4000),
#     'excl_count': np.random.randint(0, 5, 4000)
# })




# 1. 读取数据 (模拟读取您的csv，请替换为 pd.read_csv('your_file.csv'))
# 假设您的 dataframe 叫 df
try:
    df = pd.read_csv(r'D:\Projects\DTS402\dataset\merged_dataset_with_features.csv')
    # 简单的空值处理，防止报错
    df['text'] = df['text'].fillna('')
except Exception as e:
    print(f"读取文件失败，将使用模拟数据运行: {e}")

# --- 为了演示，用几条模拟数据，您运行时请删掉这一块 ---
# data = {
#     'text': [
#                 "Eva is on her to find a way to escape",
#                 "This is my favorite product ever",
#                 "I absolutely hate this verify bad",
#                 "escape the bad product",
#                 "find a way to love"
#             ] * 800  # 扩充到4000条以模拟真实计算量
# }
# df = pd.DataFrame(data)
# ----------------------------------------------------

print("正在处理数据，请稍候...")


# ==========================================
# 第一步：构建词表与计算 TF (Term Frequency)
# ==========================================

def clean_text(text):
    # 简单清洗：转小写，去除非字母字符
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()


# 1.1 分词并构建语料库
docs_tokens = df['text'].apply(clean_text).tolist()

# 1.2 统计所有词的出现次数，限制词表大小以防止内存溢出
# 建议取最高频的 2000-5000 个词，否则 4000x20000 的矩阵 SVD 会很慢
vocab_size = 3000
all_words = [word for doc in docs_tokens for word in doc]
most_common_words = [word for word, count in Counter(all_words).most_common(vocab_size)]

# 创建 词 -> 索引 的映射
word_to_index = {word: i for i, word in enumerate(most_common_words)}
n_docs = len(docs_tokens)
n_vocab = len(word_to_index)

print(f"文档数量: {n_docs}, 词表大小: {n_vocab}")

# 1.3 构建 TF 矩阵
# 初始化零矩阵 (文档数 x 词表大小)
tf_matrix = np.zeros((n_docs, n_vocab))

# 记录每个词在多少个文档中出现过 (用于计算 IDF)
doc_freq = np.zeros(n_vocab)

for doc_idx, tokens in enumerate(docs_tokens):
    doc_word_counts = Counter(tokens)
    # 记录当前文档的总词数
    total_words = len(tokens)
    if total_words == 0: continue

    seen_words_in_doc = set()

    for word, count in doc_word_counts.items():
        if word in word_to_index:
            word_idx = word_to_index[word]
            # 计算 TF: 词频 / 文档总词数
            tf_matrix[doc_idx, word_idx] = count / total_words

            seen_words_in_doc.add(word_idx)

    # 更新文档频率
    for idx in seen_words_in_doc:
        doc_freq[idx] += 1

# ==========================================
# 第二步：计算 IDF 并生成 TF-IDF 矩阵
# ==========================================

# IDF = log(文档总数 / (包含该词的文档数 + 1))
# +1 是为了平滑，防止除以0
idf_vector = np.log(n_docs / (doc_freq + 1))

# TF-IDF = TF * IDF (利用 NumPy 的广播机制)
tfidf_matrix = tf_matrix * idf_vector

print(f"TF-IDF 矩阵构建完成，形状: {tfidf_matrix.shape}")

# ==========================================
# 第三步：使用 SVD 降维到 24 维
# ==========================================

# 您的目标维度
k = 24

# 使用 numpy.linalg.svd 进行奇异值分解
# 矩阵 M = U * Sigma * Vt
# 降维后的矩阵 = U[:, :k] * Sigma[:k]
# 注意：full_matrices=False 会让计算更快
print("正在进行 SVD 降维 (这可能需要几秒钟)...")
U, Sigma, Vt = np.linalg.svd(tfidf_matrix, full_matrices=False)

# 提取前 k 个特征
U_k = U[:, :k]
Sigma_k = np.diag(Sigma[:k])

# 计算降维后的矩阵 (4000 x 24)
reduced_matrix = np.dot(U_k, Sigma_k)

# ==========================================
# 第四步：合并结果
# ==========================================

# 创建列名
col_names = [f'tfidf_svd_{i}' for i in range(k)]
df_tfidf_24 = pd.DataFrame(reduced_matrix, columns=col_names)

# 合并回原始数据
final_df = pd.concat([df, df_tfidf_24], axis=1)

print("降维完成！前5行预览：")
print(final_df.iloc[:, -24:].head())  # 只打印新生成的24列

# 如果需要，可以将 final_df 保存
final_df.to_csv(r'D:\Projects\DTS402\dataset\processed_data_with_embeddings.csv', index=False)