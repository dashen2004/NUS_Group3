import pandas as pd

# Step 1: Read the CSV file and print column names
data = pd.read_csv('athlete_data.csv')
print(data.columns)  # 打印出所有的列名
