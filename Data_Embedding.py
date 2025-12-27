# 第六步 (修正版): 独立进行 TF-IDF 和 SVD 处理
# The sixth step (Corrected): TF-IDF and SVD (Fit on Train, Transform Test)
import pandas as pd
import numpy as np
import re
from collections import Counter
import pickle  # 用于保存 IDF 和 SVD 参数

# 1. 读取切分好的数据
df_train = pd.read_csv(r'D:\Projects\DTS402\dataset\train_raw.csv')
df_test = pd.read_csv(r'D:\Projects\DTS402\dataset\test_raw.csv')

# 再次确保文本列为字符串
df_train['text'] = df_train['text'].fillna('').astype(str)
df_test['text'] = df_test['text'].fillna('').astype(str)

print("正在处理特征，这严格遵守了防止数据泄露的流程...")


# ==========================================
# 阶段 1: 构建词表 (仅基于训练集!)
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()


# 1.1 只用训练集建立 Corpus
train_docs_tokens = df_train['text'].apply(clean_text).tolist()

vocab_size = 3000
all_words = [word for doc in train_docs_tokens for word in doc]
# 选出训练集中最常见的 3000 个词
most_common_words = [word for word, count in Counter(all_words).most_common(vocab_size)]
word_to_index = {word: i for i, word in enumerate(most_common_words)}
n_vocab = len(word_to_index)

print(f"词表构建完成 (仅基于训练集)，词表大小: {n_vocab}")


# ==========================================
# 阶段 2: 计算 TF-IDF
# ==========================================

def compute_tf_matrix(docs_tokens, word_to_idx, n_v):
    """辅助函数：计算 TF 矩阵"""
    n_d = len(docs_tokens)
    tf_mat = np.zeros((n_d, n_v))
    # 用于计算 IDF 的文档频率统计 (只在处理训练集时需要，但在函数里先返回)
    doc_freqs = np.zeros(n_v)

    for i, tokens in enumerate(docs_tokens):
        counts = Counter(tokens)
        total = len(tokens)
        if total == 0: continue

        seen = set()
        for w, c in counts.items():
            if w in word_to_idx:
                idx = word_to_idx[w]
                tf_mat[i, idx] = c / total
                seen.add(idx)

        for idx in seen:
            doc_freqs[idx] += 1

    return tf_mat, doc_freqs


# 2.1 处理训练集 (Train)
print("正在计算训练集 TF-IDF...")
tf_train, doc_freq_train = compute_tf_matrix(train_docs_tokens, word_to_index, n_vocab)

# 计算 IDF (仅使用训练集的文档分布)
n_train = len(df_train)
idf_vector = np.log(n_train / (doc_freq_train + 1))  # +1 平滑处理

# 得到训练集 TF-IDF
tfidf_train = tf_train * idf_vector

# 2.2 处理测试集 (Test) - 关键点！
# 注意：测试集必须使用训练集的 word_to_index 和 idf_vector
print("正在转换测试集 TF-IDF (使用训练集的规则)...")
test_docs_tokens = df_test['text'].apply(clean_text).tolist()
# 计算测试集 TF (这里返回的 doc_freq_test 我们不需要，因为 IDF 已经定死了)
tf_test, _ = compute_tf_matrix(test_docs_tokens, word_to_index, n_vocab)

# 使用训练集的 IDF 乘以 测试集的 TF
tfidf_test = tf_test * idf_vector

# ==========================================
# 阶段 3: SVD 降维
# ==========================================
k = 24
print(f"正在进行 SVD 降维 (k={k})...")

# 3.1 在训练集上 Fit (分解)
# 训练集矩阵 M = U * Sigma * Vt
U, Sigma, Vt = np.linalg.svd(tfidf_train, full_matrices=False)

# 获取投影矩阵 (即 Vt 的前 k 行)
# 这一步非常关键：Vt 包含了“特征向量”或者说“语义方向”
Vt_k = Vt[:k, :]  # 形状 (k, n_vocab)

# 训练集的降维结果 (直接用 U * Sigma 比较快，数学上等价于 Project)
X_train_svd = np.dot(U[:, :k], np.diag(Sigma[:k]))

# 3.2 在测试集上 Transform (投影)
# 测试集不能重新做 SVD，必须投影到训练集学到的语义空间中
# 公式: Reduced_Test = Test_TFIDF * (Vt_k).T
X_test_svd = np.dot(tfidf_test, Vt_k.T)

print(f"训练集 SVD 形状: {X_train_svd.shape}")
print(f"测试集 SVD 形状: {X_test_svd.shape}")

# ==========================================
# 阶段 4: 保存最终结果
# ==========================================
# 创建列名
col_names = [f'tfidf_svd_{i}' for i in range(k)]

# 4.1 合并训练集
df_train_svd = pd.DataFrame(X_train_svd, columns=col_names)
final_train = pd.concat([df_train, df_train_svd], axis=1)

# 4.2 合并测试集
df_test_svd = pd.DataFrame(X_test_svd, columns=col_names)
final_test = pd.concat([df_test, df_test_svd], axis=1)

# 保存文件
train_out_path = r'D:\Projects\DTS402\dataset\train_processed_final.csv'
test_out_path = r'D:\Projects\DTS402\dataset\test_processed_final.csv'

final_train.to_csv(train_out_path, index=False)
final_test.to_csv(test_out_path, index=False)

print("\n✅ 处理完成！")
print(f"最终训练集已保存: {train_out_path}")
print(f"最终测试集已保存: {test_out_path}")
print("现在，训练数据和测试数据在数学上是完全隔离的，数据泄露问题已解决。")