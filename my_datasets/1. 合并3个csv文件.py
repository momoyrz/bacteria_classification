import pandas as pd
import glob

# 获取所有CSV文件的路径列表
csv_files = glob.glob('/home/ubuntu/qujunlong/data/tryy/*.csv')
csv_dir = '/home/ubuntu/qujunlong/data/tryy'

# 初始化一个空的数据框
combined_df = pd.DataFrame()

# 遍历所有CSV文件并将其合并到一个数据框中
for file in csv_files:
    df = pd.read_csv(file)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# 将合并后的数据框保存为一个新的CSV文件
combined_df.to_csv('/home/ubuntu/qujunlong/data/tryy/all_data.csv', index=False)

print("CSV files have been successfully combined.")
