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

PLOT_DIR = Path("plots_new_new_new_n ")
SAMPLE_RATE = 1.0  # 如果内存有限，可把每个文件按比例抽样（0-1），1.0 表示不抽样
RESAMPLE_SEC = 1   # 时间序列聚合的秒级窗口（用于消息率、注入率等）
TOP_N_IDS = 20     # 绘制最常见的 N 个 CAN ID

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

# def plot_injection_ratio_over_time(df, name, resample_sec=RESAMPLE_SEC):
#     """
#     绘制注入消息占比随时间变化曲线，自动调整宽度。
#     """
#     # === 数据准备 ===
#     grouped = (
#         df[['Flag']]
#         .resample(f"{resample_sec}s")['Flag']
#         .apply(lambda x: (x == 'T').sum())
#         .rename('injected')
#     )
#     total = (
#         df['CAN_ID']
#         .resample(f"{resample_sec}s")
#         .count()
#         .rename('total')
#     )
#     ratio = (grouped / total).fillna(0)
#
#     # === 自动调整图宽 ===
#     # 比如每小时 1 英寸，最多 30 英寸（防止过大）
#     time_span = (df.index.max() - df.index.min()).total_seconds()
#     hours = time_span / 3600 if not pd.isna(time_span) else 1
#     width = min(max(12, hours * 2), 30)  # 在 [12, 30] 范围内自适应
#     height = 5
#
#     # === 绘图 ===
#     plt.figure(figsize=(width, height))
#     ratio.plot(color='tab:blue', linewidth=1.2)
#     plt.title(f"{name} - injected ratio per {resample_sec}s", fontsize=13)
#     plt.ylabel("injected ratio", fontsize=11)
#     plt.xlabel("time", fontsize=11)
#     plt.grid(True, linestyle='--', alpha=0.5)
#
#     # 避免横轴刻度过密
#     plt.xticks(rotation=30, ha='right')
#
#     plt.tight_layout()
#     p = PLOT_DIR / f"{name}_injected_ratio_per{resample_sec}s.png"
#     plt.savefig(p, dpi=200)
#     plt.close()
#     print(f"✅ 保存: {p}（宽度 {width:.1f} 英寸）")

# def plot_injection_ratio_over_time(df, name, resample_sec=RESAMPLE_SEC, segment_minutes=10):
#     """
#     绘制注入消息占比随时间变化曲线。
#     按 segment_minutes 分段绘制多张图（最后不足一段也单独成图）。
#     """
#
#     # === 数据准备 ===
#     grouped = (
#         df[['Flag']]
#         .resample(f"{resample_sec}s")['Flag']
#         .apply(lambda x: (x == 'T').sum())
#         .rename('injected')
#     )
#     total = (
#         df['CAN_ID']
#         .resample(f"{resample_sec}s")
#         .count()
#         .rename('total')
#     )
#     ratio = (grouped / total).fillna(0)
#
#     # === 时间范围划分 ===
#     start_time = ratio.index.min()
#     end_time = ratio.index.max()
#     segment_delta = pd.Timedelta(minutes=segment_minutes)
#
#     if pd.isna(start_time) or pd.isna(end_time):
#         print(f"⚠️ {name} 没有有效时间索引，跳过绘图。")
#         return
#
#     current_start = start_time
#     segment_index = 1
#
#     while current_start < end_time:
#         current_end = min(current_start + segment_delta, end_time)
#         seg_ratio = ratio.loc[current_start:current_end]
#
#         if seg_ratio.empty:
#             current_start = current_end
#             continue
#
#         # === 绘图 ===
#         plt.figure(figsize=(12, 5))
#         seg_ratio.plot(color='tab:blue', linewidth=1.2)
#         plt.title(
#             f"{name} - injected ratio per {resample_sec}s\n"
#             f"{current_start.strftime('%H:%M:%S')} - {current_end.strftime('%H:%M:%S')}",
#             fontsize=13
#         )
#         plt.ylabel("injected ratio", fontsize=11)
#         plt.xlabel("time", fontsize=11)
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.xticks(rotation=30, ha='right')
#
#         plt.tight_layout()
#
#         # === 保存 ===
#         p = PLOT_DIR / f"{name}_injected_ratio_{segment_index:02d}_per{resample_sec}s.png"
#         plt.savefig(p, dpi=200)
#         plt.close()
#
#         print(f"✅ 保存分段图 {segment_index}: {p} ({current_start.strftime('%H:%M:%S')} → {current_end.strftime('%H:%M:%S')})")
#
#         # 下一个时间段
#         segment_index += 1
#         current_start = current_end



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

# def plot_interarrival_hist(df, name, sample_ids=None):
#     # 全局报文到达间隔分布（秒）
#     df_sorted = df.sort_index()
#     # compute deltas in seconds
#     timestamps = df_sorted['Timestamp'].values
#     deltas = np.diff(timestamps)
#     plt.figure(figsize=(8,4))
#     plt.hist(deltas[deltas>0], bins=200, log=True)
#     plt.title(f"{name} - inter-arrival times (log y)")
#     plt.xlabel("seconds"); plt.ylabel("count (log)")
#     plt.tight_layout()
#     p = PLOT_DIR / f"{name}_interarrival_hist.png"
#     plt.savefig(p); plt.close()
#     print(f"保存: {p}")
#
#     # 对若干热门 ID 分别绘制间隔分布
#     top_ids = list(df['CAN_ID'].value_counts().head(5).index) if sample_ids is None else sample_ids
#     for cid in top_ids:
#         sub = df[df['CAN_ID']==cid].sort_index()
#         ts = sub['Timestamp'].values
#         if len(ts) < 2:
#             continue
#         deltas = np.diff(ts)
#         plt.figure(figsize=(6,3))
#         plt.hist(deltas[deltas>0], bins=100)
#         plt.title(f"{name} - interarrival for ID {cid} (n={len(ts)})")
#         plt.xlabel("seconds"); plt.tight_layout()
#         p = PLOT_DIR / f"{name}_interarrival_id_{cid}.png"
#         plt.savefig(p); plt.close()
def plot_interarrival_hist(df, name, sample_ids=None, max_interval=1.0):
    """
    绘制报文到达间隔分布。
    自动去除超过 max_interval 秒的极端长间隔，防止横轴被拉伸。
    """
    df_sorted = df.sort_index()
    timestamps = df_sorted['Timestamp'].values
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 0]

    # === 限制横轴范围 ===
    filtered = deltas[deltas <= max_interval]
    if len(filtered) == 0:
        print(f"⚠️ {name}: 所有间隔都超过 {max_interval}s，跳过绘图。")
        return

    plt.figure(figsize=(8,4))
    plt.hist(filtered, bins=200, log=True)
    plt.title(f"{name} - inter-arrival times (<= {max_interval}s, log y)")
    plt.xlabel("interval (s)")
    plt.ylabel("count (log)")
    plt.tight_layout()
    p = PLOT_DIR / f"{name}_interarrival_hist.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"✅ 保存: {p} （已过滤 >{max_interval}s 的极端值）")

    # === 各 ID 子图 ===
    top_ids = list(df['CAN_ID'].value_counts().head(5).index) if sample_ids is None else sample_ids
    for cid in top_ids:
        sub = df[df['CAN_ID']==cid].sort_index()
        ts = sub['Timestamp'].values
        if len(ts) < 2:
            continue
        deltas = np.diff(ts)
        deltas = deltas[(deltas > 0) & (deltas <= max_interval)]
        if len(deltas) == 0:
            continue
        plt.figure(figsize=(6,3))
        plt.hist(deltas, bins=100)
        plt.title(f"{name} - ID {cid} interarrival (<= {max_interval}s)")
        plt.xlabel("seconds")
        plt.tight_layout()
        p = PLOT_DIR / f"{name}_interarrival_id_{cid}.png"
        plt.savefig(p, dpi=150)
        plt.close()


def plot_data_bytes_heatmap(df, name, cid=None, window_seconds=10):
    # 对指定 CAN ID，在时间轴上绘制 DATA0~DATA7 的 heatmap（值 0-255）
    if cid is None:
        # 选择最活跃的 ID
        cid = df['CAN_ID'].value_counts().idxmax()
    sub = df[df['CAN_ID']==cid].copy()
    if sub.empty:
        print(f"{name}: ID {cid} 没有数据，跳过 heatmap")
        return
    # 按较小时间窗口聚合（mean）
    agg = sub[['DATA0','DATA1','DATA2','DATA3','DATA4','DATA5','DATA6','DATA7']].resample(f"{window_seconds}s").mean()
    if agg.dropna(how='all').empty:
        print(f"{name}: 聚合后无数据，跳过 heatmap")
        return
    plt.figure(figsize=(12,4))
    sns.heatmap(agg.T, cbar=True)
    plt.title(f"{name} - data bytes heatmap for CAN ID {cid} (window {window_seconds}s)")
    plt.ylabel("byte index"); plt.xlabel("time-window index")
    plt.tight_layout()
    p = PLOT_DIR / f"{name}_heatmap_id_{cid}.png"
    plt.savefig(p); plt.close()
    print(f"保存: {p}")


# def plot_data_bytes_lines(df, name, cid=None, window_seconds=10):
#     """
#     对指定 CAN ID 绘制 DATA0~DATA7 随时间变化的折线图。
#     每个数据字节一条曲线，8 种颜色。
#     """
#     if cid is None:
#         cid = df['CAN_ID'].value_counts().idxmax()
#
#     sub = df[df['CAN_ID'] == cid].copy()
#     if sub.empty:
#         print(f"{name}: ID {cid} 没有数据，跳过绘图")
#         return
#
#     # 按时间窗口聚合平均值
#     agg = (
#         sub[['DATA0','DATA1','DATA2','DATA3','DATA4','DATA5','DATA6','DATA7']]
#         .resample(f"{window_seconds}s")
#         .mean()
#     )
#     if agg.dropna(how='all').empty:
#         print(f"{name}: 聚合后无数据，跳过绘图")
#         return
#
#     # === 绘制折线图 ===
#     plt.figure(figsize=(12, 5))
#     colors = plt.cm.tab10.colors  # 8 种常用颜色
#     for i, col in enumerate(agg.columns):
#         plt.plot(agg.index, agg[col], label=col, color=colors[i % len(colors)], linewidth=1.2)
#
#     plt.title(f"{name} - DATA0~7 over time for CAN ID {cid} (window {window_seconds}s)", fontsize=13)
#     plt.xlabel("Time")
#     plt.ylabel("Byte value (0–255)")
#     plt.ylim(0, 255)
#     plt.grid(True, linestyle='--', alpha=0.4)
#     plt.legend(ncol=4, fontsize=9, loc='upper right')
#     plt.tight_layout()
#
#     p = PLOT_DIR / f"{name}_data_bytes_lines_id_{cid}.png"
#     plt.savefig(p, dpi=200)
#     plt.close()
#     print(f"✅ 保存: {p} （DATA0~7 折线图）")

def plot_data_bytes_lines(df, name, cid=None, window_seconds=10):
    """
    对指定 CAN ID 绘制 DATA0~DATA7 随时间变化的折线图：
    1️⃣ 一张总览图（8 条线叠加）
    2️⃣ 每条线单独一张图，保持与总图相同坐标比例
    """
    if cid is None:
        cid = df['CAN_ID'].value_counts().idxmax()

    sub = df[df['CAN_ID'] == cid].copy()
    if sub.empty:
        print(f"{name}: ID {cid} 没有数据，跳过绘图")
        return

    # ===== 按时间窗口聚合平均值 =====
    agg = (
        sub[['DATA0','DATA1','DATA2','DATA3','DATA4','DATA5','DATA6','DATA7']]
        .resample(f"{window_seconds}s")
        .mean()
    )

    if agg.dropna(how='all').empty:
        print(f"{name}: 聚合后无数据，跳过绘图")
        return

    # === 总览图 ===
    plt.figure(figsize=(12, 5))
    colors = plt.cm.tab10.colors  # 8 种颜色
    for i, col in enumerate(agg.columns):
        plt.plot(agg.index, agg[col], label=col, color=colors[i % len(colors)], linewidth=1.2)

    plt.title(f"{name} - DATA0~7 over time for CAN ID {cid} (window {window_seconds}s)", fontsize=13)
    plt.xlabel("Time")
    plt.ylabel("Byte value (0–255)")
    plt.ylim(0, 255)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(ncol=4, fontsize=9, loc='upper right')
    plt.tight_layout()

    p_total = PLOT_DIR / f"{name}_data_bytes_lines_id_{cid}_ALL.png"
    plt.savefig(p_total, dpi=200)
    plt.close()
    print(f"✅ 保存: {p_total}（总览图）")

    # === 单独绘制每个 DATAi ===
    for i, col in enumerate(agg.columns):
        plt.figure(figsize=(12, 3))
        plt.plot(agg.index, agg[col], color=colors[i % len(colors)], linewidth=1.5)
        plt.title(f"{name} - {col} over time for CAN ID {cid} (window {window_seconds}s)", fontsize=12)
        plt.xlabel("Time")
        plt.ylabel("Byte value (0–255)")
        plt.ylim(0, 255)  # 与总览图统一比例
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        p_single = PLOT_DIR / f"{name}_data_byte_{col}_id_{cid}.png"
        plt.savefig(p_single, dpi=200)
        plt.close()
        print(f"  └─ 保存单独曲线图: {p_single}")


def plot_id_time_scatter(df, name, top_n=1000):
    # 散点：时间 vs CAN ID（ID 转为排序索引以便展示）
    vc = df['CAN_ID'].value_counts()
    top_ids = vc.head(200).index  # 取前 200 id 来绘图
    sub = df[df['CAN_ID'].isin(top_ids)].copy()
    if sub.empty:
        return
    # map ID to ordinal
    id_map = {cid:i for i,cid in enumerate(sorted(top_ids))}
    sub['id_ord'] = sub['CAN_ID'].map(id_map)
    plt.figure(figsize=(12,6))
    plt.scatter(sub['Timestamp'], sub['id_ord'], s=1)
    plt.title(f"{name} - time vs CAN ID (top {len(top_ids)} IDs)")
    plt.xlabel("timestamp (s)"); plt.ylabel("CAN ID ordinal")
    plt.tight_layout()
    p = PLOT_DIR / f"{name}_time_vs_id_scatter.png"
    plt.savefig(p); plt.close()
    print(f"保存: {p}")

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
    plot_top_ids(df, name)
    plot_interarrival_hist(df, name)
    # heatmap for most active ID
    top_id = df['CAN_ID'].value_counts().idxmax()
    # plot_data_bytes_heatmap(df, name, cid=top_id)
    plot_data_bytes_lines(df, name, cid=top_id)
    plot_id_time_scatter(df, name)
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
    # 如果想把多个文件合并做总体视图：
    if processed:
        print("\n正在合并所有数据（用于总体统计，如 global top IDs）...")
        all_df = pd.concat(processed.values(), ignore_index=False).sort_index()
        # 全局热图：最常见的 30 ID 的每秒消息率矩阵（示例）
        ensure_plot_dir()
        top_ids = all_df['CAN_ID'].value_counts().head(30).index
        # todo 新版本pandas适应
        # pivot = all_df[all_df['CAN_ID'].isin(top_ids)]['CAN_ID'].resample(f"{RESAMPLE_SEC}S").apply(lambda x: x.value_counts()).unstack(fill_value=0)
        tmp = all_df[all_df['CAN_ID'].isin(top_ids)].copy()
        # 以时间和 CAN_ID 双重分组统计每秒的出现次数
        pivot = (
            tmp
            .groupby([pd.Grouper(freq=f"{RESAMPLE_SEC}s"), "CAN_ID"])
            .size()
            .unstack(fill_value=0)
        )
        # pivot 可能非常大；这里只保存前三十 id 的简版热图（按时间）
        plt.figure(figsize=(12,6))
        sns.heatmap(pivot.T, cmap='viridis')
        plt.title("Global - per-second counts for top 30 CAN IDs (rows=CAN ID, cols=time-window)")
        plt.tight_layout()
        p = PLOT_DIR / f"global_top30_ids_time_heatmap.png"
        plt.savefig(p); plt.close()
        print(f"保存: {p}")
        # 也保存全局 top IDs bar
        plt.figure(figsize=(10,8))
        all_df['CAN_ID'].value_counts().head(50).plot(kind='barh')
        plt.title("Global - top 50 CAN IDs")
        plt.tight_layout()
        p = PLOT_DIR / f"global_top50_ids.png"
        plt.savefig(p); plt.close()
        print(f"保存: {p}")

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
