import pandas as pd

# 1. 定义文件路径
input_path = 'D:/Projects/DTS402/dataset/FRDDS_1k.csv'
output_path = 'D:/Projects/DTS402/dataset/FRDDS_1k_cleaned.csv'

# 2. 读取数据
# 如果读取时报错，尝试加 encoding='utf-8' 或 encoding='latin1'
df = pd.read_csv(input_path)

# --- 核心步骤：选择并重排数据列 ---
target_columns = ['category', 'rating', 'label', 'text']

try:
    df_cleaned = df[target_columns].copy()  # 加上 .copy() 防止警告

    # 【新增步骤】文本清洗：解决乱码隐患
    # 1. 确保是字符串
    df_cleaned['text'] = df_cleaned['text'].fillna('').astype(str)

    # 2. 替换弯引号为直引号 (这一步最关键)
    df_cleaned['text'] = df_cleaned['text'].str.replace('’', "'", regex=False)

    # 3. 保存文件
    # 【关键修改】encoding='utf-8-sig'
    # 加了 -sig 后，Excel 打开就不会乱码了
    df_cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"处理成功！")
    print(f"文件已保存至: {output_path}")

    # 打印一条包含引号的数据看看效果
    print("预览（检查引号是否正常）:")
    print(df_cleaned[df_cleaned['text'].str.contains("'")].head(1)['text'].values)

except KeyError as e:
    print(f"错误：缺少列 {e}")