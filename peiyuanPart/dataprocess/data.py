import os
import pandas as pd

# 获取指定目录下的所有文件，并排序
def get_sorted_files_from_data_dir(directory):
    onlyfiles = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.csv')]
    onlyfiles.sort()  # 按文件名排序
    return [os.path.join(directory, f) for f in onlyfiles]

# 合并多个CSV文件
def merge_data():
    data_dir = './'  # 这里的路径根据实际情况调整
    files = get_sorted_files_from_data_dir(data_dir)
    print("Found files:", files)

    # 合并所有文件
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    return df

# 处理数据，包括去除缺失值、日期处理等
def process_data(df):
    # 检查是否有缺失值
    missing_values = df.isnull().sum()
    print("Missing values:", missing_values)

    # 去除缺失地址的记录
    df = df.dropna(subset=['Address'])

    # 处理日期列，假设日期列为 'Incident Date'
    df['dates_processed'] = pd.to_datetime(df['Incident Date'], format='%B %d, %Y')

    # 按日期严格降序排序
    df = df.sort_values(by='dates_processed', ascending=False)

    return df

# 主函数
def main():
    # 合并数据
    df = merge_data()

    # 处理数据
    df = process_data(df)

    # 可选：将处理后的数据保存到新的CSV文件
    df.to_csv('ProcessedData_Sorted.csv', index=False)
    print("Data processed and saved as 'ProcessedData_Sorted.csv'.")

# 调用主函数
if __name__ == "__main__":
    main()
