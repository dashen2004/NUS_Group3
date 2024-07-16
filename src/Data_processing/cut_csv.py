import pandas as pd

def extract_columns(in_file, out_file, num_columns=11):
    # 读取CSV文件
    df = pd.read_csv(in_file)

    # 提取所需列
    df_extracted = pd.concat([df.iloc[:, :num_columns], df.iloc[:, -2:]], axis=1)

    # 重命名前11列为F1, F2, F3, ..., F11，最后两列为L1, L2
    column_names = [f'F{i+1}' for i in range(num_columns)] + ['L1', 'L2']
    df_extracted.columns = column_names

    # 打乱数据
    df_shuffled = df_extracted.sample(frac=1).reset_index(drop=True)

    # 保存提取后的数据到新CSV文件
    df_shuffled.to_csv(out_file, index=False)


input_file = '../../database/generate_data/all_data_processed4_pro.csv'
output_file = '../../database/model_data/normalize_data1.csv'

extract_columns(input_file, output_file)
