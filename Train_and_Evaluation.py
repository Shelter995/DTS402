# ==========================================
# 4. 训练与评估 (Training & Evaluation)
# ==========================================
import numpy as np

from Data_Embedding_demo import final_df
from Model import SimpleMLP
from Split import X_train, y_train, X_test, y_test

# 4.1 初始化模型
# input_size: 特征数量 (比如 24个SVD特征 + 评分 + 字数 = 26左右)
input_dim = X_train.shape[1]
model = SimpleMLP(input_size=input_dim, hidden_size=16, output_size=1, learning_rate=0.1)

# 4.2 开始训练
print("开始训练神经网络...")
history = model.train(X_train, y_train, epochs=2000)

# 4.3 预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# 4.4 手写评估指标 (不能用 sklearn.metrics)
def calculate_metrics(y_true, y_pred, dataset_name="Test"):
    # 展平数组
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # 基础计算
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # 指标公式
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"--- {dataset_name} Set Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


calculate_metrics(y_train, y_pred_train, "Training")
calculate_metrics(y_test, y_pred_test, "Test")


import matplotlib.pyplot as plt
import seaborn as sns # 如果没有安装seaborn，可以只用matplotlib，但seaborn更好看
# 如果报错 "No module named 'seaborn'"，请把 import seaborn 注释掉，不影响运行

# 设置绘图风格
plt.style.use('default')

# ==============================================================================
# 图表 1: 数据类别分布 (Class Balance)
# 对应作业要求: "Dataset Comparison... Class balance" [cite: 66]
# ==============================================================================
plt.figure(figsize=(8, 5))
label_counts = final_df['label'].value_counts()
colors = ['#4c72b0', '#55a868'] # 蓝色和绿色

bars = plt.bar(label_counts.index, label_counts.values, color=colors, alpha=0.8)
plt.title('Class Distribution: Original (OR) vs Computer Generated (CG)', fontsize=14)
plt.ylabel('Number of Samples')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 在柱状图上显示具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=12)
plt.savefig(r'D:\Projects\DTS402\output\1_class_distribution.png', dpi=300, bbox_inches='tight')
print("已保存: 1_class_distribution.png")
plt.show()

# ==============================================================================
# 图表 2: SVD 特征散点图 (2D Visualization)
# 对应作业要求: "Feature distributions... visualizations" [cite: 65]
# 展示前两个维度是否能将两类数据分开
# ==============================================================================
plt.figure(figsize=(10, 6))

# 提取前两个 SVD 特征
svd_x = final_df['tfidf_svd_0']
svd_y = final_df['tfidf_svd_1']
labels = final_df['label']

# 绘制散点
for label_name, color in zip(['OR', 'CG'], colors):
    mask = labels == label_name
    plt.scatter(svd_x[mask], svd_y[mask],
                c=color, label=label_name, alpha=0.6, s=20, edgecolor='w')

plt.title('Visualization of Top 2 TF-IDF SVD Components', fontsize=14)
plt.xlabel('SVD Dimension 1 (tfidf_svd_0)')
plt.ylabel('SVD Dimension 2 (tfidf_svd_1)')
plt.legend(title='Label')
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig(r'D:\Projects\DTS402\output\2_svd_visualization.png', dpi=300, bbox_inches='tight')
print("已保存: 2_svd_visualization.png")
plt.show()

# ==============================================================================
# 图表 3: 训练 Loss 曲线 (Learning Curve)
# 对应作业要求: "Model Training and Reporting... figures" [cite: 91]
# 为了画图，我们需要重新简单跑一遍训练来记录 Loss 历史
# ==============================================================================
print("正在生成 Loss 曲线数据...")
loss_history = []

plt.figure(figsize=(10, 5))

# 直接使用 history 变量
plt.plot(range(len(history)), history, color='#c44e52', linewidth=2)

plt.title('MLP Training Loss Curve', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Binary Cross-Entropy Loss')
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig(r'D:\Projects\DTS402\output\3_loss_curve.png', dpi=300, bbox_inches='tight')
print("已保存: 3_loss_curve.png")
plt.show()


# ==============================================================================
# 图表 4: 混淆矩阵 (Confusion Matrix)
# 对应作业要求: "Evaluation Metrics... predictive accuracy" [cite: 95]
# 这是一个纯 Matplotlib 手写的混淆矩阵热力图
# ==============================================================================
# 1. 计算混淆矩阵数值
y_pred_final = model.predict(X_test).flatten()
y_true_final = y_test.flatten()

tp = np.sum((y_true_final == 1) & (y_pred_final == 1))
tn = np.sum((y_true_final == 0) & (y_pred_final == 0))
fp = np.sum((y_true_final == 0) & (y_pred_final == 1))
fn = np.sum((y_true_final == 1) & (y_pred_final == 0))

cm = np.array([[tn, fp], [fn, tp]]) # 标准格式: [[TN, FP], [FN, TP]]

# 2. 绘图
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set)', fontsize=14)
plt.colorbar()

# 设置标签
classes = ['Fake (CG)', 'Original (OR)']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0)
plt.yticks(tick_marks, classes)

# 在格子里填数字
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center", fontsize=16,
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(r'D:\Projects\DTS402\output\4_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("已保存: 4_confusion_matrix.png")
plt.show()