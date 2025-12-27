# 第九步: 训练评估与可视化 (Evaluation & Visualization)
# ==========================================
# 4. Training & Evaluation
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from Model import SimpleMLP
from TrainingData_Prepare import X_train, y_train, X_test, y_test, feature_cols

# 确保输出目录存在
output_dir = r'D:\Projects\DTS402\output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ---------------------------------------------------------
# ✅ 修正点：实例化模型
# ---------------------------------------------------------
print("Initializing model...")
input_dim = X_train.shape[1]
# 重新创建一个新的模型实例，确保从头开始训练
model = SimpleMLP(input_size=input_dim, learning_rate=0.001, momentum=0.9, reg_lambda=0.001)

# 4.1 开始训练 (Start Training)
print("Start training the neural network...")
# ✅ 修正点：使用 model.train 而不是 SimpleMLP.train
history = model.train(X_train, y_train, X_test, y_test, epochs=2000, batch_size=32)

# 4.2 预测 (Prediction)
# ✅ 修正点：使用 model.predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# 4.3 打印评估指标 (Metrics)
def calculate_metrics(y_true, y_pred, dataset_name="Test"):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"--- {dataset_name} Set Metrics ---\n")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\n")


calculate_metrics(y_train, y_pred_train, "Training")
calculate_metrics(y_test, y_pred_test, "Test")

# Set the drawing style
plt.style.use('default')

# ==============================================================================
# Chart 1: Data Class Balance Distribution
# ==============================================================================
temp_df = pd.DataFrame({'label': np.where(y_train.flatten() == 1, 'OR', 'CG')})
temp_df_test = pd.DataFrame({'label': np.where(y_test.flatten() == 1, 'OR', 'CG')})
combined_labels = pd.concat([temp_df, temp_df_test])

plt.figure(figsize=(8, 5))
label_counts = combined_labels['label'].value_counts()
colors = ['#4c72b0', '#55a868']

bars = plt.bar(label_counts.index, label_counts.values, color=colors, alpha=0.8)
plt.title('Class Distribution: Original (OR) vs Computer Generated (CG)', fontsize=14)
plt.ylabel('Number of Samples')
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=12)
save_path_1 = os.path.join(output_dir, '1_class_distribution.png')
plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path_1}")
plt.show()

# ==============================================================================
# Chart 2: SVD Feature Scatter Plot
# ==============================================================================
# 找到 tfidf_svd_0 和 tfidf_svd_1 在特征列表中的位置
idx_svd0 = feature_cols.index('tfidf_svd_0')
idx_svd1 = feature_cols.index('tfidf_svd_1')

print(f"修正索引: tfidf_svd_0 在第 {idx_svd0} 列, tfidf_svd_1 在第 {idx_svd1} 列")

plt.figure(figsize=(10, 6))

# 使用正确的索引提取数据
svd_x = X_train[:, idx_svd0]
svd_y = X_train[:, idx_svd1]

# 这里的 y_train 是 (N, 1) 的数组，展平用于绘图
labels_flat = y_train.flatten()

# 绘制散点
# OR (1) -> Blue, CG (0) -> Green
plt.scatter(svd_x[labels_flat == 1], svd_y[labels_flat == 1],
            c='#4c72b0', label='OR', alpha=0.6, s=20, edgecolor='w')
plt.scatter(svd_x[labels_flat == 0], svd_y[labels_flat == 0],
            c='#55a868', label='CG', alpha=0.6, s=20, edgecolor='w')

plt.title('Visualization of Top 2 TF-IDF SVD Components', fontsize=14)
plt.xlabel('SVD Dimension 1 (tfidf_svd_0)')
plt.ylabel('SVD Dimension 2 (tfidf_svd_1)')
plt.legend(title='Label')
plt.grid(True, linestyle='--', alpha=0.3)

save_path_2 = os.path.join(output_dir, '2_svd_visualization.png')
plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path_2}")
plt.show()

# ==============================================================================
# Chart 3: Loss Curve
# ==============================================================================
plt.figure(figsize=(10, 5))
plt.plot(history['loss'], color='#c44e52', linewidth=2)
plt.title('MLP Training Loss Curve', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Binary Cross-Entropy Loss')
plt.grid(True, linestyle='--', alpha=0.5)
save_path_3 = os.path.join(output_dir, '3_loss_curve.png')
plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path_3}")
plt.show()

# ==============================================================================
# Chart 4: Precision Curve
# ==============================================================================
plt.figure(figsize=(10, 5))
plt.plot(history['train_precision'], color='#4c72b0', linewidth=2, label='Train Precision')
plt.plot(history['val_precision'], color='#55a868', linewidth=2, linestyle='--', label='Validation Precision')
plt.title('Precision Curve: Train vs Validation', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Precision Score')
plt.legend()
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
save_path_4 = os.path.join(output_dir, '4_precision_curve.png')
plt.savefig(save_path_4, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path_4}")
plt.show()

# ==============================================================================
# Chart 5: Confusion Matrix (Best Model)
# ==============================================================================
print("The best model (model_best.pkl) is being loaded for the final evaluation...")
best_model_path = r"D:\Projects\DTS402\output\model\model_best.pkl"
try:
    model.load_model(best_model_path)

    y_pred_final = model.predict(X_test).flatten()
    y_true_final = y_test.flatten()

    tp = np.sum((y_true_final == 1) & (y_pred_final == 1))
    tn = np.sum((y_true_final == 0) & (y_pred_final == 0))
    fp = np.sum((y_true_final == 0) & (y_pred_final == 1))
    fn = np.sum((y_true_final == 1) & (y_pred_final == 0))
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Best Model)', fontsize=14)
    plt.colorbar()

    classes = ['Fake (CG)', 'Original (OR)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center", fontsize=16,
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    save_path_5 = os.path.join(output_dir, '5_confusion_matrix_best.png')
    plt.savefig(save_path_5, dpi=300, bbox_inches='tight')
    plt.show()
except FileNotFoundError:
    print(f"Warning: Best model file not found at {best_model_path}. Skipping confusion matrix plot.")