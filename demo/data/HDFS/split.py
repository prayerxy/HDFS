import pandas as pd
import csv  # 导入 csv 模块
# 定义输入文件名和输出文件名的模板
input_file = 'HDFS_100k.log_structured.csv'  # 替换成你的输入文件名
output_template = 'HDFS_100k.log_structured_{}.csv'  # 输出文件名的模板，{} 将被替换为编号

# 读取输入文件
df = pd.read_csv(input_file)

# 计算每份的行数
total_rows = len(df)
rows_per_file = total_rows // 5  # 每份的行数，整数除法，舍弃余数

# 划分数据并保存为CSV文件
for i in range(5):
    start_index = i * rows_per_file
    end_index = (i + 1) * rows_per_file if i < 4 else total_rows  # 最后一份取剩余的所有行

    # 提取子数据框
    subset_df = df.iloc[start_index:end_index]

    # 构建输出文件名
    output_file = output_template.format(i + 1)

    # 保存为CSV文件，保持原格式
    subset_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

print("数据已经成功分成了五份，并保存为CSV文件。")
