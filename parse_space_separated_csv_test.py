# import pandas as pd
#
#
# def smart_read_car_csv(path):
#     # 先读一行看看分隔符
#     with open(path, 'r', encoding='utf-8') as f:
#         first_line = f.readline()
#
#     if ',' in first_line:
#         sep = ','
#     elif '\t' in first_line:
#         sep = '\t'
#     else:
#         sep = r'\s+'
#
#     df = pd.read_csv(path,
#                      sep=sep,
#                      header=None,
#                      names=["Timestamp", "CAN_ID", "DLC",
#                             "DATA0", "DATA1", "DATA2", "DATA3",
#                             "DATA4", "DATA5", "DATA6", "DATA7", "Flag"],
#                      engine="python")
#     return df
#
#
# df = smart_read_car_csv("DoS_dataset.csv")
# print(df.head())
import pandas as pd

df = pd.read_csv("DoS_dataset.csv",
                 sep=',',
                 header=None,
                 names=["Timestamp", "CAN_ID", "DLC",
                        "DATA0","DATA1","DATA2","DATA3",
                        "DATA4","DATA5","DATA6","DATA7","Flag"])

pd.set_option('display.max_columns', None)  # 显示所有列

print(df.tail(20))
print("列数：", df.shape[1])
