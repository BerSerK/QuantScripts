import pandas as pd
import os

# 定义基本路径
base_path = 'netvalue/20241008'

# 定义文件名和输出文件名
file_names = {
    '5亿CTA.xlsx': '5yi',
    '10亿CTA.xlsx': '10yi'
}

def process_data_no_header(file_name, output_name):
    file_path = os.path.join(base_path, file_name)
    # 读取Excel文件
    data = pd.read_excel(file_path)

    # 格式化日期列为 'YYYY/MM/DD' 并设置为第一列
    data['日期'] = pd.to_datetime(data['日期']).dt.strftime('%Y/%m/%d')

    # 计算单位净值偏差量（减去1），设置为第二列
    data['单位净值偏差量'] = data['单位净值'] - 1

    # 选择需要的列并重命名
    data = data[['日期', '单位净值偏差量']]
    data.columns = ['Date', 'Net Value Deviation']

    # 保存到CSV文件，不包括表头
    output_csv_path = os.path.join(base_path, f'{output_name}.csv')
    data.to_csv(output_csv_path, index=False, header=False)
    return output_csv_path

# 处理数据并删除原始文件
for file_name, output_name in file_names.items():
    output_csv_path = process_data_no_header(file_name, output_name)
    # 删除原始Excel文件
    os.remove(os.path.join(base_path, file_name))

# 确认文件夹内容
files_remaining = os.listdir(base_path)
files_remaining
