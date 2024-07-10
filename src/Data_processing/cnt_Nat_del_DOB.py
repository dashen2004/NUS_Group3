import pandas as pd

if __name__ == "__main__":

    # 读取CSV文件
    file_path = '../../docs/new_data_s2i.csv'  # 请将其替换为你的文件路径
    df = pd.read_csv(file_path)

    # 计算每个Nat的平均Results Score
    nat_avg_scores = df.groupby('Nat')['Results Score'].mean()

    # 将平均分数保存到另一个CSV文件
    nat_avg_scores_df = nat_avg_scores.reset_index()
    nat_avg_scores_df.columns = ['Nat', 'Score']
    nat_avg_scores_file_path = '../../docs/mapping_nat_avg_scores.csv'
    nat_avg_scores_df.to_csv(nat_avg_scores_file_path, index=False)

    # 将Nat列替换为对应的平均Results Score
    df['Nat'] = df['Nat'].map(nat_avg_scores)

    # 删除DOB列
    df = df.drop(columns=['DOB'])

    # 将结果写入新的CSV文件
    new_file_path = '../../docs/new_data_n2c.csv'
    df.to_csv(new_file_path, index=False)

    print("Complete")
