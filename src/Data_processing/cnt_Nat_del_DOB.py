import pandas as pd

if __name__ == "__main__":

    # 读取CSV文件
    file_path = '../../database/generate_data/new_data_s2i.csv'
    # 如果取最好，可以改为nation5_data.csv
    score_path = '../../database/crawl_data/sorted_athlete_sprint_data_with_event_ages_revised.csv'
    df = pd.read_csv(file_path)
    sf = pd.read_csv(score_path)

    # 计算每个Nat的平均Results Score
    nat_avg_scores = sf.groupby('Nat')['Results Score'].mean()

    # 将平均分数保存到另一个CSV文件
    nat_avg_scores_df = nat_avg_scores.reset_index()
    nat_avg_scores_df.columns = ['Nat', 'Score']
    nat_avg_scores_file_path = '../../database/generate_data/mapping_nat_avg_scores.csv'
    nat_avg_scores_df.to_csv(nat_avg_scores_file_path, index=False)

    # 将Nat列替换为对应的平均Results Score
    df['Nat'] = df['Nat'].map(nat_avg_scores)

    # 删除DOB列
    df = df.drop(columns=['DOB'])

    # 将结果写入新的CSV文件
    new_file_path = '../../database/generate_data/new_data_n2c.csv'
    df.to_csv(new_file_path, index=False)

    print("Complete")
