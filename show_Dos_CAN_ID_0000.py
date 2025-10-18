import pandas as pd

# === 读取数据 ===
df = pd.read_csv("DoS_dataset.csv",
                 sep=',',
                 header=None,
                 names=["Timestamp", "CAN_ID", "DLC",
                        "DATA0","DATA1","DATA2","DATA3",
                        "DATA4","DATA5","DATA6","DATA7","Flag"])

# === 显示所有列 ===
pd.set_option('display.max_columns', None)

# === 筛选 CAN_ID 为 "0000" 的行 ===
df_0000 = df[df["CAN_ID"] == "0000"]

# === 再筛选 DATA0~DATA7 中至少有一个不是 "00" 的 ===
data_cols = [f"DATA{i}" for i in range(8)]
mask_nonzero = (df_0000[data_cols] != "00").any(axis=1)
df_0000_nonzero = df_0000[mask_nonzero]

# === 打印前 20 行 ===
print(df_0000_nonzero.head(20))

# === 输出统计信息 ===
print(f"\n共有 {len(df_0000_nonzero)} 条 CAN_ID=0000 且 DATA 含非00 的记录")
print("列数：", df.shape[1])
