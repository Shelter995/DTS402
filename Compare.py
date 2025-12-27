# 第九步: 最终多数据集对比评估
# The ninth step: Evaluate different datasets using the best model.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import Counter

from Data_Embedding import clean_text, compute_tf_matrix, word_to_index, n_vocab, idf_vector, Vt_k
from Model import SimpleMLP
from TrainingData_Prepare import X_train, X_mean, X_std, X_test, y_test, y_train

print("=== 开始最终评估 (Final Evaluation) ===")

# ==========================================
# 0. 准备环境与模型
# ==========================================
# 重新实例化模型结构 (3层)
input_dim = X_train.shape[1]
model_best = SimpleMLP(input_size=input_dim, learning_rate=0.001, momentum=0.9, reg_lambda=0.001)

# 加载训练好的最佳权重
best_model_path = r"D:\Projects\DTS402\output\model\model_best.pkl"
if os.path.exists(best_model_path):
    model_best.load_model(best_model_path)
    print(f"✅ 已加载最佳模型: {best_model_path}")
else:
    print("⚠️ 未找到最佳模型文件，使用当前内存中的模型。")


# ==========================================
# 1. 定义数据转换管道 (Transform Pipeline)
# ==========================================
# 这是一个核心函数：它模拟了"将新数据放入已训练好的系统中"的过程
# 它必须严格使用训练集(Cell 6)学到的参数：word_to_index, idf_vector, Vt_k, X_mean, X_std
def transform_pipeline(df_raw, dataset_name):
    print(f"\n正在处理 {dataset_name} ({len(df_raw)} 条样本)...")

    # 1. 文本预处理
    df_raw['text'] = df_raw['text'].fillna('').astype(str)

    # -----------------------------------------------------------
    # 修正点 1: 提取 Rating 列
    # 训练集中包含了 rating，所以这里也必须有，否则维度对不上 (26 vs 27)
    # -----------------------------------------------------------
    if 'rating' in df_raw.columns:
        rating = df_raw['rating'].values.reshape(-1, 1)
    else:
        print("   ⚠️ 警告: 缺少 'rating' 列，使用 0 填充 (可能影响精度)")
        rating = np.zeros((len(df_raw), 1))

    # 2. 提取手动特征
    word_count = df_raw['text'].apply(lambda x: len(x.split())).values.reshape(-1, 1)
    excl_count = df_raw['text'].apply(lambda x: x.count('!')).values.reshape(-1, 1)

    # 3. 提取 SVD 特征
    tokens_list = df_raw['text'].apply(clean_text).tolist()
    # 使用 Cell 6 的参数计算 TF-IDF 和 SVD
    tf_mat, _ = compute_tf_matrix(tokens_list, word_to_index, n_vocab)
    tfidf_mat = tf_mat * idf_vector
    svd_mat = np.dot(tfidf_mat, Vt_k.T)

    # -----------------------------------------------------------
    # 修正点 2: 拼接所有特征 (注意顺序)
    # 假设训练集加载顺序是: [rating, word_count, excl_count, svd_0...svd_23]
    # -----------------------------------------------------------
    # 我们先尝试把 rating 放在最前面
    X_combined = np.hstack([rating, word_count, excl_count, svd_mat])

    # 维度检查 (Debug用)
    if X_combined.shape[1] != X_mean.shape[0]:
        # 如果还是对不上，可能是 rating 在训练集中并不是第一列
        # 但通常 pandas 读取 csv 会保持顺序。如果报错，我们再调整
        print(f"   ⚠️ 维度警告: 生成特征 {X_combined.shape[1]} vs 训练集特征 {X_mean.shape[0]}")

    # 5. 标准化
    try:
        X_final = (X_combined - X_mean) / X_std
    except ValueError as e:
        print(f"   ❌ 标准化失败，维度不匹配: {e}")
        return None, None

    # 6. 准备标签
    y_final = np.where(df_raw['label'] == 'OR', 1, 0).reshape(-1, 1)

    return X_final, y_final


# ==========================================
# 2. 读取并处理三个数据集
# ==========================================
results = []


# 定义评估函数
def evaluate_and_record(name, X, y):
    preds = model_best.predict(X)
    tp = np.sum((y == 1) & (preds == 1))
    tn = np.sum((y == 0) & (preds == 0))
    fp = np.sum((y == 0) & (preds == 1))
    fn = np.sum((y == 1) & (preds == 0))

    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

    results.append({
        'name': name, 'samples': len(y),
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1
    })
    print(f"   -> [结果] F1: {f1:.4f}, Acc: {acc:.4f}")


# --- 数据集 A: Open Dataset (原始文件) ---
path_open = r'D:\Projects\DTS402\dataset\open_3k.csv'
if os.path.exists(path_open):
    df_open = pd.read_csv(path_open)
    X_open, y_open = transform_pipeline(df_open, "Open Dataset (3k)")
    evaluate_and_record("Open Dataset", X_open, y_open)
else:
    print(f"❌ 找不到文件: {path_open}")

# --- 数据集 B: Self Dataset (原始文件) ---
path_self = r'D:\Projects\DTS402\dataset\self_dataset_cleaned.csv'
if os.path.exists(path_self):
    df_self = pd.read_csv(path_self)
    X_self, y_self = transform_pipeline(df_self, "Self Dataset")
    evaluate_and_record("Self Dataset", X_self, y_self)
else:
    print(f"❌ 找不到文件: {path_self}")

# --- 数据集 C: Mixed Dataset (Training + Test) ---
# 这个我们直接用内存中的数据拼接，不需要重新读取文件
# 因为 processed_data_with_embeddings.csv 本质上就是 X_train 和 X_test 的合集
print("\n正在处理 Mixed Dataset (All Data)...")
X_total = np.vstack([X_train, X_test])
y_total = np.vstack([y_train, y_test])
evaluate_and_record("Mixed (Total)", X_total, y_total)

# ==========================================
# 3. 可视化结果
# ==========================================
print("\n生成对比图表...")
if len(results) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 图 1: 柱状图
    names = [r['name'] for r in results]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    keys = ['accuracy', 'precision', 'recall', 'f1']

    x = np.arange(len(names))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, key in enumerate(keys):
        vals = [r[key] for r in results]
        axes[0].bar(x + i * width, vals, width, label=metrics[i], color=colors[i], alpha=0.85)

    axes[0].set_title('Model Performance Across Datasets', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Score')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(names)
    axes[0].legend(loc='lower right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, 1.1)

    # 图 2: 饼图 (样本量)
    samples = [r['samples'] for r in results]
    axes[1].pie(samples, labels=names, autopct='%1.1f%%',
                colors=['#3498db', '#9b59b6', '#34495e'], startangle=140)
    axes[1].set_title('Dataset Sample Sizes', fontsize=14)

    plt.tight_layout()
    save_path = r'D:\Projects\DTS402\output\6_final_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存: {save_path}")
    plt.show()

# ==========================================
# 4. 打印详细表格
# ==========================================
print("\n" + "=" * 75)
print(f"{'Dataset Name':<20} {'Samples':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 75)
for r in results:
    print(
        f"{r['name']:<20} {r['samples']:<10} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}")
print("=" * 75)