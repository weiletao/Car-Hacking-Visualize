import re
import numpy as np
import pandas as pd
import random

def parse_txt_format_debug(path):
    """
    调试版 normal_run_data.txt 解析器。
    会打印匹配统计信息，并随机展示原始行与解析结果。
    """
    rows = []
    total_lines = 0
    match_stats = {
        "timestamp": 0,
        "id": 0,
        "dlc": 0,
        "data": 0
    }
    bad_lines = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_lines += 1
            raw = line.strip()
            if not raw:
                continue

            # Timestamp
            m_ts = re.search(r"Timestamp:\s*([0-9]+\.[0-9]+|[0-9]+)", raw)
            ts = float(m_ts.group(1)) if m_ts else np.nan
            if m_ts:
                match_stats["timestamp"] += 1

            # CAN ID
            m_id = re.search(r"ID:\s*([0-9a-fA-F]{1,4})", raw)
            cid = m_id.group(1).lower() if m_id else None
            if m_id:
                match_stats["id"] += 1

            # DLC
            m_dlc = re.search(r"DLC:\s*([0-9])", raw)
            dlc = int(m_dlc.group(1)) if m_dlc else np.nan
            if m_dlc:
                match_stats["dlc"] += 1

            # DATA bytes
            m_data = re.search(r"DLC:\s*[0-9]\s*(.*)$", raw)
            data_bytes = []
            if m_data:
                tail = m_data.group(1).strip()
                parts = re.split(r'\s+', tail)
                for p in parts:
                    if re.fullmatch(r'[0-9a-fA-F]{1,2}', p):
                        try:
                            data_bytes.append(int(p, 16))
                        except:
                            data_bytes.append(np.nan)
                if len(data_bytes) > 0:
                    match_stats["data"] += 1

            # pad to 8 bytes
            while len(data_bytes) < 8:
                data_bytes.append(np.nan)

            # Flag 默认 R
            flag = "R"

            rows.append([ts, cid, dlc] + data_bytes + [flag])

            # 检测异常
            if not m_ts or not m_id or not m_dlc:
                bad_lines.append(raw)

    print(f"\n--- 解析统计 ---")
    print(f"总行数: {total_lines}")
    print(f"成功匹配 Timestamp: {match_stats['timestamp']} 行")
    print(f"成功匹配 CAN_ID: {match_stats['id']} 行")
    print(f"成功匹配 DLC: {match_stats['dlc']} 行")
    print(f"成功匹配 DATA: {match_stats['data']} 行")
    print(f"总解析结果行数: {len(rows)}")
    print(f"异常行（缺失关键字段）数量: {len(bad_lines)}")

    # 打印几个异常样本
    if len(bad_lines) > 0:
        print("\n--- 异常样本（前 5 行） ---")
        for l in bad_lines[:5]:
            print(l)

    # 转成 DataFrame
    col_names = ["Timestamp", "CAN_ID", "DLC"] + [f"DATA{i}" for i in range(8)] + ["Flag"]
    df = pd.DataFrame(rows, columns=col_names)

    # 检查 Timestamp 转换后的 NaN 数量
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    nan_ts = df["Timestamp"].isna().sum()
    print(f"\n有效 Timestamp: {len(df) - nan_ts}, NaN 数量: {nan_ts}")

    # 随机展示 5 行解析结果
    print("\n--- 随机抽样 5 行解析结果 ---")
    print(df.sample(min(5, len(df))))

    return df


if __name__ == "__main__":
    path = "normal_run_data.txt"
    df = parse_txt_format_debug(path)
    print(f"\n最终 DataFrame 行数: {len(df)}")
    print(f"列数: {df.shape[1]}")
