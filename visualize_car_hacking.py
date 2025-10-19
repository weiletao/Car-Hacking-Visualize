#!/usr/bin/env python3
"""
visualize_car_hacking.py

目标：读取 Car-Hacking Dataset 提供的 CSV / TXT 文件，生成一组时间序列与统计可视化图表，保存在 ./plots/ 下。

依赖：
    pip install pandas matplotlib numpy seaborn tqdm

用法：
    把本脚本放在数据文件同一目录，或通过 --data-dir 指定目录，运行：
    python visualize_car_hacking.py

输出：plots/ 目录下的 PNG 文件
"""

# import os
import re
import argparse
from pathlib import Path
# from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")  # 仅用于美观（可删）

# ---- 参数 ----
DEFAULT_FILES = [
    "DoS_dataset.csv",
    "Fuzzy_dataset.csv",
    "gear_dataset.csv",
    "RPM_dataset.csv",
    "normal_run_data.txt",
]

PLOT_DIR = Path("plots")
SAMPLE_RATE = 1.0  # 如果内存有限，可把每个文件按比例抽样（0-1），1.0 表示不抽样
RESAMPLE_SEC = 10   # 时间序列聚合的秒级窗口（用于消息率、注入率等）
TOP_N_IDS = 20     # 绘制最常见的 N 个 CAN ID
TIME_WINDOW_SEC = 1

# ---- 解析函数 ----
def parse_space_separated_csv(path):
    """
    智能解析 car-hacking 数据集文件（兼容空格 / 制表符 / 逗号分隔）
    支持字段数不固定（DLC<8）场景，自动补齐列。
    """

    col_names = ["Timestamp", "CAN_ID", "DLC",
                 "DATA0","DATA1","DATA2","DATA3",
                 "DATA4","DATA5","DATA6","DATA7","Flag"]

    # ---- 自动检测分隔符 ----
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()

    if ',' in first_line:
        sep = ','
    elif '\t' in first_line:
        sep = '\t'
    else:
        sep = r'\s+'

    # ---- 用 pandas 读取 ----
    try:
        df = pd.read_csv(path,
                         sep=sep,
                         header=None,
                         names=col_names,
                         engine="python")
    except Exception as e:
        print(f"[警告] 读取 {path} 时出错：{e}")
        print("回退到手动逐行解析模式（兼容格式错误）...")
        # 回退方案（逐行 split）
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = re.split(r'\s+|,|\t', line)
                if len(parts) < 12:
                    parts += [None] * (12 - len(parts))
                elif len(parts) > 12:
                    parts = parts[:12]
                rows.append(parts)
        df = pd.DataFrame(rows, columns=col_names)

    # ---- 字段类型转换 ----
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["CAN_ID"] = df["CAN_ID"].astype(str).str.lower()
    df["Flag"] = df["Flag"].astype(str).str.upper()

    # ---- DATA[0~7] 转换为十进制整数 ----
    for i in range(8):
        col = f"DATA{i}"
        df[col] = df[col].apply(
            lambda x: int(x, 16)
            if isinstance(x, str) and re.fullmatch(r'[0-9a-fA-F]{1,2}', x)
            else np.nan
        )

    return df

def parse_txt_format(path):
    """
    解析 normal_run_data.txt 这种行：
    Timestamp: 1479121434.850202        ID: 0350    000    DLC: 8    05 28 84 66 6d 00 00 a2
    返回统一列格式 DataFrame
    """
    rows = []
    invalid_lines = 0  # 记录无效行数量（缺失关键字段）

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 提取 Timestamp
            m_ts = re.search(r"Timestamp:\s*([0-9]+\.[0-9]+|[0-9]+)", line)
            ts = float(m_ts.group(1)) if m_ts else np.nan

            # 提取 CAN ID
            m_id = re.search(r"ID:\s*([0-9a-fA-F]{1,4})", line)
            cid = m_id.group(1).lower() if m_id else None

            # 提取 DLC
            m_dlc = re.search(r"DLC:\s*([0-9])", line)
            dlc = int(m_dlc.group(1)) if m_dlc else np.nan

            # 提取 DATA 字节（8 个 hex）
            data_bytes = []
            m_data = re.search(r"DLC:\s*[0-9]\s*(.*)$", line)
            if m_data:
                tail = m_data.group(1).strip()
                parts = re.split(r'\s+', tail)
                for p in parts:
                    if re.fullmatch(r'[0-9a-fA-F]{1,2}', p):
                        try:
                            data_bytes.append(int(p, 16))
                        except Exception:
                            data_bytes.append(np.nan)

            # pad to 8
            while len(data_bytes) < 8:
                data_bytes.append(np.nan)

            # Flag (txt 中没有 T/R 标记，默认为 R)
            flag = 'R'

            # --- 关键改进：跳过无效行 ---
            # 如果 CAN_ID 或 DLC 缺失，则认为该行无效（仅 Timestamp）
            if cid is None or np.isnan(dlc):
                invalid_lines += 1
                continue

            rows.append([ts, cid, dlc] + data_bytes + [flag])

    col_names = ["Timestamp", "CAN_ID", "DLC"] + [f"DATA{i}" for i in range(8)] + ["Flag"]
    df = pd.DataFrame(rows, columns=col_names)

    # 类型清理
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["CAN_ID"] = df["CAN_ID"].astype(str).str.lower()
    df["Flag"] = df["Flag"].astype(str).str.upper()

    print(f"✅ 解析完成: 共 {len(df)} 条有效记录，跳过 {invalid_lines} 条无效行。")

    return df

# ---- 聚合与绘图 ----
def ensure_plot_dir():
    PLOT_DIR.mkdir(exist_ok=True)

def basic_stats_and_time_index(df, name):
    print(f"\n--- {name} 基本信息 ---")
    total = len(df)
    n_inj = df['Flag'].value_counts().get('T', 0)
    n_norm = df['Flag'].value_counts().get('R', 0)
    first_ts = df['Timestamp'].min()
    last_ts = df['Timestamp'].max()
    duration = (last_ts - first_ts) if pd.notna(first_ts) and pd.notna(last_ts) else np.nan
    print(f"总消息数: {total:,}, 注入(T): {n_inj:,}, 正常(R): {n_norm:,}")
    print(f"时间范围: {first_ts} 到 {last_ts} （持续 {duration:.2f} 秒）")
    # 时间索引
    df = df.sort_values("Timestamp").reset_index(drop=True)
    # make datetime index for resampling convenience
    df['dt'] = pd.to_datetime(df['Timestamp'], unit='s', origin='unix')
    df = df.set_index('dt')
    return df

def plot_message_rate(df, name, resample_sec=RESAMPLE_SEC):
    # 每秒消息数（或更粗的时间窗）
    s = df['CAN_ID'].resample(f"{resample_sec}s").count()
    # # 直接统计行数呢？两个结果是相同的
    # s = df.resample(f"{resample_sec}s").size()

    plt.figure(figsize=(12,4))
    s.plot()
    plt.title(f"{name} - message rate (per {resample_sec}s)")
    plt.xlabel("time"); plt.ylabel("messages")
    plt.tight_layout()
    p = PLOT_DIR / f"{name}_message_rate_per{resample_sec}s.png"
    plt.savefig(p); plt.close()
    print(f"保存: {p}")

def plot_injection_ratio_over_time(df, name, resample_sec=RESAMPLE_SEC):
    # 注入消息占比随时间变化
    grouped = df[['Flag']].resample(f"{resample_sec}s")['Flag'].apply(lambda x: (x=='T').sum()).rename('injected')
    total = df['CAN_ID'].resample(f"{resample_sec}s").count().rename('total')
    ratio = (grouped / total).fillna(0)
    plt.figure(figsize=(12,4))
    ratio.plot()
    plt.title(f"{name} - injected ratio per {resample_sec}s")
    plt.ylabel("injected ratio")
    plt.xlabel("time")
    plt.tight_layout()
    p = PLOT_DIR / f"{name}_injected_ratio_per{resample_sec}s.png"
    plt.savefig(p); plt.close()
    print(f"保存: {p}")

def plot_rate_and_ratio(df, name, resample_sec=RESAMPLE_SEC):
    """
    在同一张图上绘制：
      - 每秒消息数（蓝色）
      - 注入消息比例（红色，右侧 Y 轴）
    共享时间轴，便于比较攻击注入与总流量的关系。
    """
    # === 数据准备 ===
    # 每秒消息总数
    total = df.resample(f"{resample_sec}s").size().rename("total_msgs")

    # 每秒注入消息数量
    injected = df[['Flag']].resample(f"{resample_sec}s")['Flag'].apply(lambda x: (x == 'T').sum()).rename("injected")

    # 注入比例
    ratio = (injected / total).fillna(0)

    # === 绘图 ===
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(total.index, total.values, color='tab:blue', label='Message Rate')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Messages per window", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 第二个 Y 轴（右侧）
    ax2 = ax1.twinx()
    ax2.plot(ratio.index, ratio.values, color='tab:red', linestyle='--', label='Injection Ratio')
    ax2.set_ylabel("Injection Ratio", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 图标题和布局
    plt.title(f"{name} - Message Rate & Injection Ratio (per {resample_sec}s)")
    fig.tight_layout()

    # 图例（同时显示两条线的标签）
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    # 保存
    p = PLOT_DIR / f"{name}_rate_and_ratio_per{resample_sec}s.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"✅ 保存: {p}")

def plot_rate_and_ratio_normalized(df, name, resample_sec=RESAMPLE_SEC):
    """
    绘制 归一化后的消息速率 与 注入比例 曲线（同一坐标轴）。
    便于比较二者的相对变化趋势。
    """
    # === 数据准备 ===
    total = df.resample(f"{resample_sec}s").size().rename("total_msgs")
    injected = df[['Flag']].resample(f"{resample_sec}s")['Flag'].apply(lambda x: (x == 'T').sum()).rename("injected")
    ratio = (injected / total).fillna(0)

    # --- 归一化 ---
    total_norm = (total - total.min()) / (total.max() - total.min() + 1e-9)
    ratio_norm = (ratio - ratio.min()) / (ratio.max() - ratio.min() + 1e-9)

    # === 绘图 ===
    plt.figure(figsize=(12, 5))
    plt.plot(total_norm.index, total_norm.values, color='tab:blue', label='Message Rate (normalized)')
    plt.plot(ratio_norm.index, ratio_norm.values, color='tab:red', linestyle='--', label='Injection Ratio (normalized)')
    plt.title(f"{name} - Normalized Message Rate & Injection Ratio (per {resample_sec}s)", fontsize=13)
    plt.xlabel("Time")
    plt.ylabel("Normalized Value (0–1)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # === 保存 ===
    p = PLOT_DIR / f"{name}_rate_and_ratio_normalized_per{resample_sec}s.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"✅ 保存: {p}")

def plot_top_ids(df, name, top_n=TOP_N_IDS):
    top = df['CAN_ID'].value_counts().head(top_n)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top.values, y=top.index)
    plt.title(f"{name} - Top {top_n} CAN IDs by count")
    plt.xlabel("count"); plt.ylabel("CAN ID")
    plt.tight_layout()
    p = PLOT_DIR / f"{name}_top{top_n}_can_ids.png"
    plt.savefig(p); plt.close()
    print(f"保存: {p}")

def plot_data_combined_hex(df, name, cid=None, highlight_injections=True):
    """
    将 DATA0~DATA7 拼成一个 64 位整数（视作整体 payload）绘图。
    🔴 注入数据 (Flag=='T') 用红点标注。
    """
    import numpy as np

    if cid is None:
        cid = df['CAN_ID'].value_counts().idxmax()

    sub = df[df['CAN_ID'] == cid].copy()
    if sub.empty:
        print(f"{name}: ID {cid} 没有数据，跳过绘图")
        return

    # === 确保时间索引 ===
    if not isinstance(sub.index, pd.DatetimeIndex):
        if "Timestamp" in sub.columns:
            sub.index = pd.to_datetime(sub["Timestamp"], unit="s")

    # === 构造 64 位整数（假设 DATA0 是最高位，DATA7 是最低位）===
    cols = [f"DATA{i}" for i in range(8) if f"DATA{i}" in sub.columns]
    if len(cols) < 8:
        print(f"{name}: ID {cid} 数据字段不完整（{len(cols)}/8），跳过拼接。")
        return

    payload = np.zeros(len(sub), dtype=np.uint64)
    for i in range(8):
        payload = np.left_shift(payload, 8) + sub[f"DATA{i}"].astype(np.uint64)

    sub["payload"] = payload
    sub["payload_norm"] = np.log1p(payload)

    inj = sub[sub["Flag"] == "T"]
    normal = sub[sub["Flag"] != "T"]

    # === 绘图 ===
    plt.figure(figsize=(14, 5))
    plt.plot(normal.index, normal["payload_norm"], color="blue", linewidth=1.0, label="Normal (log payload)")
    if highlight_injections and not inj.empty:
        plt.scatter(inj.index, inj["payload_norm"], color="red", s=20, alpha=0.5, label="Injection")

    plt.title(f"{name} - Combined DATA0–7 (log scaled) for CAN ID {cid}", fontsize=13)
    plt.xlabel("Time")
    plt.ylabel("log(1 + payload value)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    p = PLOT_DIR / f"{name}_data_combined_hex_id_{cid}.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"✅ 保存: {p}（DATA0~7 合并曲线）")

def plot_id_timewindow_scatter(df, name, window_seconds=TIME_WINDOW_SEC, top_n=100, injection_alpha=0.3):
    """
    按时间窗口聚合的 CAN ID 散点图：
    - 分别绘制正常数据、注入数据、以及综合图。
    - 横轴：时间窗口（实际时间刻度），三幅图横轴范围统一。
    - 纵轴：CAN ID（取出现频率最高的 top_n 个）。
    """

    # === 确保索引是 DatetimeIndex ===
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Timestamp" in df.columns:
            df = df.copy()
            df.index = pd.to_datetime(df["Timestamp"], unit="s")
        else:
            print(f"{name}: 无 Timestamp 字段，无法按时间聚合")
            return

    # === 取最常出现的 top_n IDs ===
    top_ids = df["CAN_ID"].value_counts().head(top_n).index
    sub = df[df["CAN_ID"].isin(top_ids)].copy()
    if sub.empty:
        print(f"{name}: 无有效 CAN ID 数据")
        return

    # === 创建时间窗口（floor 到最近的 window_seconds）===
    sub["time_bin"] = sub.index.floor(f"{window_seconds}s")

    # === 数值化 CAN ID 以便绘图 ===
    id_map = {cid: i for i, cid in enumerate(sorted(top_ids))}
    sub["id_ord"] = sub["CAN_ID"].map(id_map)

    # === 区分注入与正常消息 ===
    inj = sub[sub["Flag"] == "T"]
    normal = sub[sub["Flag"] != "T"]

    # === 统一横轴范围 ===
    xmin = sub["time_bin"].min()
    xmax = sub["time_bin"].max()

    # -------------------------------------------------
    # ① 仅正常数据
    # -------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.scatter(normal["time_bin"], normal["id_ord"], s=3, color="blue", alpha=0.8)
    plt.title(f"{name} - Normal CAN frames (top {top_n} IDs)", fontsize=13)
    plt.xlabel("Time")
    plt.ylabel("CAN ID (ordinal index)")
    plt.yticks(range(len(top_ids)), sorted(top_ids), fontsize=8)
    plt.xlim(xmin, xmax)  # ✅ 统一横轴范围
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    p_normal = PLOT_DIR / f"{name}_id_timewindow_scatter_normal_{window_seconds}s.png"
    plt.savefig(p_normal, dpi=200)
    plt.close()
    print(f"✅ 保存: {p_normal}（正常数据）")

    # -------------------------------------------------
    # ② 仅注入数据
    # -------------------------------------------------
    if not inj.empty:
        plt.figure(figsize=(12, 6))
        plt.scatter(inj["time_bin"], inj["id_ord"], s=6, color="red", alpha=0.8)
        plt.title(f"{name} - Injection frames (top {top_n} IDs)", fontsize=13)
        plt.xlabel("Time")
        plt.ylabel("CAN ID (ordinal index)")
        plt.yticks(range(len(top_ids)), sorted(top_ids), fontsize=8)
        plt.xlim(xmin, xmax)  # ✅ 统一横轴范围
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        p_inj = PLOT_DIR / f"{name}_id_timewindow_scatter_injection_{window_seconds}s.png"
        plt.savefig(p_inj, dpi=200)
        plt.close()
        print(f"✅ 保存: {p_inj}（注入数据）")

    # -------------------------------------------------
    # ③ 综合图（蓝 + 红）
    # -------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.scatter(normal["time_bin"], normal["id_ord"], s=3, color="blue", alpha=0.8, label="Normal")
    if not inj.empty:
        plt.scatter(inj["time_bin"], inj["id_ord"], s=6, color="red", alpha=injection_alpha, label="Injection")

    plt.title(f"{name} - Normal + Injection frames (window {window_seconds}s)", fontsize=13)
    plt.xlabel("Time")
    plt.ylabel("CAN ID (ordinal index)")
    plt.yticks(range(len(top_ids)), sorted(top_ids), fontsize=8)
    plt.xlim(xmin, xmax)  # ✅ 统一横轴范围
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    p_combined = PLOT_DIR / f"{name}_id_timewindow_scatter_combined_{window_seconds}s.png"
    plt.savefig(p_combined, dpi=200)
    plt.close()
    print(f"✅ 保存: {p_combined}（综合图：蓝=正常，红=注入，alpha={injection_alpha}）")

def plot_data_bytes_heatmap_raw(df, name, cid=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    if cid is None:
        cid = df['CAN_ID'].value_counts().idxmax()

    sub = df[df['CAN_ID'] == cid].copy()
    if sub.empty:
        print(f"{name}: ID {cid} 没有数据，跳过绘图")
        return

    # ✅ 保留原始 timestamp，不转 datetime
    if "Timestamp" not in sub.columns:
        print(f"{name}: 无 Timestamp 字段，无法绘图")
        return

    # === 只取 DATA0~7 与 Timestamp ===
    cols = [f"DATA{i}" for i in range(8) if f"DATA{i}" in sub.columns]
    data = sub[cols].T  # 转置，使 DATA0~7 为行
    timestamps = sub["Timestamp"].values

    sub_orig = sub.copy()

    # # 若数据太多，可进行稀疏采样
    # if len(data.columns) > 2000:
    #     step = max(1, len(data.columns)//2000)
    #     data = data.iloc[:, ::step]
    #     timestamps = timestamps[::step]
    #     # 同步注入点稀疏
    #     sub = sub.iloc[::step, :]

    # === 画原始热力图 ===
    plt.figure(figsize=(14, 4))
    sns.heatmap(data, cmap="viridis", cbar=True, vmin=0, vmax=255)
    plt.title(f"{name} - DATA0~7 heatmap (raw timestamp, CAN ID {cid})", fontsize=12)
    plt.ylabel("Byte index (DATA0–DATA7)")
    plt.xlabel("Timestamp (s)")
    plt.xticks(
        np.linspace(0, len(timestamps)-1, 10),
        [f"{t:.2f}" for t in np.linspace(timestamps[0], timestamps[-1], 10)],
        rotation=45, ha="right"
    )
    plt.tight_layout()
    p1 = PLOT_DIR / f"{name}_heatmap_raw_timestamp_id_{cid}.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"✅ 保存: {p1}（基础热力图）")

    # === 带注入标记的版本 ===
    inj_ts = sub.loc[sub["Flag"] == "T", "Timestamp"].values
    if len(inj_ts) == 0:
        print(f"{name}: 没有注入数据，跳过注入标注图。")
        return

    plt.figure(figsize=(14, 4))
    ax = sns.heatmap(data, cmap="viridis", cbar=True, vmin=0, vmax=255)
    plt.title(f"{name} - DATA0~7 heatmap with injections (CAN ID {cid})", fontsize=12)
    plt.ylabel("Byte index (DATA0–DATA7)")
    plt.xlabel("Timestamp (s)")

    # ✅ 设置横轴刻度
    plt.xticks(
        np.linspace(0, len(timestamps)-1, 10),
        [f"{t:.2f}" for t in np.linspace(timestamps[0], timestamps[-1], 10)],
        rotation=45, ha="right"
    )

    # ✅ 在注入点处画竖线（半透明红色）
    inj_indices = np.searchsorted(timestamps, inj_ts)
    for idx in inj_indices:
        if 0 <= idx < len(timestamps):
            plt.axvline(idx, color="red", alpha=0.15, linewidth=1.2)

    plt.tight_layout()
    p2 = PLOT_DIR / f"{name}_heatmap_raw_timestamp_id_{cid}_with_injections.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"✅ 保存: {p2}（带注入标记的热力图）")

    print("注入点总数 (原始 sub):", sub_orig['Flag'].eq('T').sum())
    print("注入点数 (采样后 sub):", sub['Flag'].eq('T').sum())

# ---- 主流程 ----
def process_file(path, name, sample_rate=SAMPLE_RATE):
    print(f"\n读取 {path} ...")
    if str(path).lower().endswith(".txt"):
        df = parse_txt_format(path)
    else:
        df = parse_space_separated_csv(path)
    # 抽样（如果需要）
    if sample_rate < 1.0:
        df = df.sample(frac=sample_rate, random_state=42).reset_index(drop=True)
    # 基本清理
    df = basic_stats_and_time_index(df, name)
    # 绘图
    ensure_plot_dir()
    plot_message_rate(df, name)
    plot_injection_ratio_over_time(df, name)
    plot_rate_and_ratio(df, name)  # 双轴图（原始值）
    plot_rate_and_ratio_normalized(df, name)  # 单轴归一化图（对比趋势）
    plot_top_ids(df, name)

    top_id = df['CAN_ID'].value_counts().idxmax()
    plot_data_combined_hex(df, name, cid=top_id)
    plot_id_timewindow_scatter(df, name)

    # heatmap for most active ID
    # plot_data_bytes_heatmap_raw(df, name, cid=top_id)

    # 返回 df 以便合并/进一步分析
    return df

def main(data_dir, files):
    data_dir = Path(data_dir)
    processed = {}
    for fname in files:
        p = data_dir / fname
        if not p.exists():
            print(f"警告: 找不到 {p}, 跳过")
            continue
        name = Path(fname).stem
        df = process_file(p, name)
        processed[name] = df

    print("\n全部完成。所有图保存在 ./plots/ 目录下。")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="数据文件所在目录")
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="要处理的文件名列表")
    ap.add_argument("--sample-rate", type=float, default=SAMPLE_RATE, help="抽样比例（0-1）")
    ap.add_argument("--resample-sec", type=int, default=RESAMPLE_SEC, help="时间聚合窗口（秒）")
    args = ap.parse_args()
    SAMPLE_RATE = args.sample_rate
    RESAMPLE_SEC = args.resample_sec
    main(args.data_dir, args.files)
