import pandas as pd

# 1. 定义文件路径
file1_path = 'D:/Projects/DTS402/dataset/open_3k.csv'
file2_path = 'D:/Projects/DTS402/dataset/self_dataset_cleaned.csv' # 注意确认文件名是否完全一致
output_path = '/dataset/merged_dataset.csv'

# 2. 读取两个 CSV 文件
# 使用 utf-8-sig 确保之前的特殊字符（如弯引号）能被正确读取
df1 = pd.read_csv(file1_path, encoding='utf-8-sig')
df2 = pd.read_csv(file2_path, encoding='utf-8-sig')

print(f"文件1大小: {df1.shape}")
print(f"文件2大小: {df2.shape}")

# 3. 检查列名是否完全一致 (可选，但推荐)
if list(df1.columns) != list(df2.columns):
    print("⚠️ 警告：两个文件的列名顺序或名称不完全一致，pandas 会尝试按列名对齐合并。")
    print(f"文件1列名: {list(df1.columns)}")
    print(f"文件2列名: {list(df2.columns)}")
else:
    print("✅ 列名检查通过，格式一致。")

# 4. 合并数据
# ignore_index=True 相当于自动执行了 reset_index(drop=True)
# 它会忽略原来两个表各自的索引，重新生成 0, 1, 2, 3... 的新索引
df_merged = pd.concat([df1, df2], ignore_index=True)

print(f"合并后总大小: {df_merged.shape}")

# 5. 保存为新文件
df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"保存成功！合并文件位于: {output_path}")