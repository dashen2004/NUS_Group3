import pandas as pd
import numpy as np

file1_path = '../../database/generate_data/all_data_processed4_pro.csv'
file2_path = '../main_model/data_year_order.csv'

df = pd.read_csv(file1_path)

# 获取Year及其之后的所有列
columns_to_keep = df.columns[df.columns.get_loc('Year'):]

# 保留Year及其之后的列
df = df[columns_to_keep]

# 将所有数值列转换为浮点类型
df = df.astype({col: 'float64' for col in df.select_dtypes(include=[np.number]).columns})

# 按Year进行排序
df = df.sort_values(by='Year')

# 获取Year以后的所有列（不包括Year）
columns_to_rename = df.columns[1:]

# 创建新的列名列表
new_column_names = [f'F{i+1}' for i in range(len(columns_to_rename)-2)] + ['L1', 'L2']

# 重新命名列
df = df.rename(columns=dict(zip(columns_to_rename, new_column_names)))

# 保存处理后的数据到新的CSV文件
df.to_csv(file2_path, index=False)

print("Complete")
