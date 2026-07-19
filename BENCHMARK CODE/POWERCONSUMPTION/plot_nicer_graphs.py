#!/usr/bin/env python3
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_DIR       = "."
TIME_LIMIT    = 1200
PREFIXES      = ["PI5", "STM"]
SMOOTH_WINDOW = 15
FIG_DPI       = 200
POWER_THRESHOLD = 0.6

DEVICE_NAMES = {
    "PI5": "Raspberry Pi 5 with Hailo AI HAT +2",
    "STM": "STM32MP257FDK"
}

DETECT_REPLACE = {
    "PI5": "yolov8n:640",
    "STM": "yolov8n:320"
}

CUSTOM_PALETTE = [
    "#0077B6",
    "#2ca02c",
    "#ff7f0e",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

PERF_MAP = {
    "STM": {
        "dir": "ST_PERFORMANCES",
        "configs": [
            ("STM_detect_cpu_only", "avg_fps_cpu_yolo_solo.txt", None),
            ("STM_detect_npu_only", "avg_fps_npu_yolo_solo.txt", None),
            ("STM_functiongemma_cpu_only", None, "avg_tps_cpu_gemma_solo.txt"),
            ("STM_detect_cpu_functiongemma_cpu", "avg_fps_cpu_yolo_with_other.txt", "avg_tps_cpu_gemma_with_other.txt"),
            ("STM_detect_npu_functiongemma_cpu", "avg_fps_npu_yolo_with_other.txt", "avg_tps_cpu_gemma_with_other.txt"),
        ]
    },
    "PI5": {
        "dir": "PI5_PERFORMANCES",
        "configs": [
            ("PI5_qwen3_cpu_only", None, "avg_tps_cpu_qwen_solo.txt"),
            ("PI5_qwen3_hailo_only", None, "avg_tps_hailo_qwen_solo.txt"),
            ("PI5_detect_cpu_only", "avg_fps_cpu_yolo_solo.txt", None),
            ("PI5_detect_hailo_only", "avg_fps_hailo_yolo_solo.txt", None),
            ("PI5_qwen3_cpu_detect_hailo", "avg_fps_hailo_yolo_with_other.txt", "avg_tps_cpu_qwen_with_other.txt"),
            ("PI5_detect_cpu_qwen3_hailo", "avg_fps_cpu_yolo_with_other.txt", "avg_tps_hailo_qwen_with_other.txt"),
        ]
    }
}

plt.rcParams.update({
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.frameon":    False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

def _clean_label(name: str, family: str) -> str:
    label = name
    if label.startswith(family + "_"):
        label = label[len(family) + 1:]

    if family in DETECT_REPLACE:
        label = label.replace("detect", DETECT_REPLACE[family])

    return label.replace("_", " ")

def load_csvs(directory: str, prefix: str, time_limit: float) -> tuple[dict[str, pd.DataFrame], int, int]:
    pattern = os.path.join(directory, f"{prefix}*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARNING] No CSV files found for prefix '{prefix}' in {directory}")
        return {}, 0, 0

    datasets: dict[str, pd.DataFrame] = {}
    total_original = 0
    total_kept = 0

    for fpath in files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        df = pd.read_csv(fpath, comment="#")
        df.columns = df.columns.str.strip()
        df = df[df["Time"] <= time_limit].copy()
        df["Power"] = df["Voltage"] * df["Current"]

        original_len = len(df)
        df = df[df["Power"] >= POWER_THRESHOLD].copy()
        kept_len = len(df)
        discarded = original_len - kept_len

        total_original += original_len
        total_kept += kept_len

        datasets[name] = df
        print(f"  Loaded {name}: {kept_len} points kept ({discarded} artifacts < {POWER_THRESHOLD} W discarded)")

    return datasets, total_original, total_kept

def plot_family(datasets: dict[str, pd.DataFrame], family: str) -> None:
    if not datasets:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (name, df) in enumerate(datasets.items()):
        base_label = _clean_label(name, family)
        colour = CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)]

        mean_power = df["Power"].mean()
        std_power = df["Power"].std()
        legend_label = f"{base_label} ($\\mu$={mean_power:.2f} W, $\\sigma$={std_power:.2f} W)"

        power = (df["Power"]
                 .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
                 .mean())
        ax.plot(df["Time"], power,
                label=legend_label, linewidth=1.6, color=colour, alpha=0.88)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [W]")

    device_title = DEVICE_NAMES.get(family, family)
    ax.set_title(f"Power Consumption - {device_title}", fontweight="bold", pad=12)

    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.18, linewidth=0.5)

    ax.legend(fontsize=9,
              loc="center left",
              bbox_to_anchor=(1.01, 0.5),
              borderaxespad=0)

    fig.subplots_adjust(left=0.09, right=0.62, top=0.90, bottom=0.12)
    out = f"power_consumption_{family}.png"
    fig.savefig(out, dpi=FIG_DPI)
    print(f"  Saved plot -> {out}")
    plt.close(fig)

def build_metric_table(all_datasets: dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    rows = []
    for name, df in all_datasets.items():
        rows.append({
            "Configuration": name,
            "Min": df[metric].min(),
            "Mean": df[metric].mean(),
            "Max": df[metric].max(),
            "Std Dev": df[metric].std(),
        })
    return pd.DataFrame(rows).set_index("Configuration")

def save_metric_latex(df: pd.DataFrame, metric: str, path: str) -> None:
    df.index = df.index.str.replace("_", " ")

    unit_map = {"Current": "A", "Voltage": "V", "Power": "W"}
    unit = unit_map.get(metric, "")

    df.columns = [f"{col} [{unit}]" for col in df.columns]

    latex = df.to_latex(
        float_format="%.4f",
        caption=f"{metric} statistics per configuration.",
        label=f"tab:{metric.lower()}_stats",
        column_format="lcccc",
    )

    latex = latex.replace(r"\begin{table}", r"\begin{table}[H]")

    with open(path, "w") as f:
        f.write(latex)
    print(f"  Saved table -> {path}")

def read_perf_file(filepath: str, key: str) -> str:
    if not os.path.exists(filepath):
        print(f"  [WARNING] Performance file not found: {filepath}")
        return "N/A"

    with open(filepath, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == key:
                    try:
                        return f"{float(v.strip()):.2f}"
                    except ValueError:
                        return v.strip()

    print(f"  [WARNING] Key '{key}' not found in {filepath}")
    return "N/A"

def build_performance_table(base_dir: str) -> pd.DataFrame:
    rows = []
    for family, info in PERF_MAP.items():
        perf_dir = os.path.join(base_dir, info["dir"])
        for name, fps_file, tps_file in info["configs"]:
            fps_val = "NO" if fps_file is None else read_perf_file(os.path.join(perf_dir, fps_file), "avg_fps")
            tps_val = "NO" if tps_file is None else read_perf_file(os.path.join(perf_dir, tps_file), "generation_only_tok_per_s")
            rows.append({
                "Configuration": name,
                "Avg FPS": fps_val,
                "Avg TPS": tps_val
            })

    return pd.DataFrame(rows).set_index("Configuration")

def save_perf_latex(df: pd.DataFrame, path: str) -> None:
    df.index = df.index.str.replace("_", " ")

    latex = df.to_latex(
        caption="Measured performance (FPS and TPS) per configuration.",
        label="tab:measured_performances",
        column_format="lcc",
    )

    latex = latex.replace(r"\begin{table}", r"\begin{table}[H]")

    with open(path, "w") as f:
        f.write(latex)
    print(f"  Saved table -> {path}")

def main():
    csv_dir = sys.argv[1] if len(sys.argv) > 1 else CSV_DIR

    all_datasets: dict[str, pd.DataFrame] = {}

    for prefix in PREFIXES:
        print(f"\n[{prefix}] Loading CSVs from '{csv_dir}' …")
        family, total_original, total_kept = load_csvs(csv_dir, prefix, TIME_LIMIT)
        all_datasets.update(family)

        if total_original > 0:
            discarded_count = total_original - total_kept
            discard_pct = (discarded_count / total_original) * 100
            print(f"\n  [ARTIFACT REJECTION REPORT - {prefix}]")
            print(f"  Threshold applied: Power < {POWER_THRESHOLD} W")
            print(f"  Total original samples: {total_original}")
            print(f"  Discarded samples:      {discarded_count}")
            print(f"  Percentage discarded:    {discard_pct:.4f}%\n")

        print(f"[{prefix}] Plotting …")
        plot_family(family, prefix)

    if not all_datasets:
        print("\nNo data loaded - nothing to do.")
        return

    print("\n[TABLE] Building power statistics …")
    for metric in ["Current", "Voltage", "Power"]:
        stats_df = build_metric_table(all_datasets, metric)
        print(f"\n--- {metric} Stats ---")
        print(stats_df.to_string())
        save_metric_latex(stats_df, metric, f"{metric.lower()}_stats.tex")

    print("\n[TABLE] Building performance table …")
    perf_df = build_performance_table(csv_dir)
    print(perf_df.to_string())
    save_perf_latex(perf_df, "measured_performances.tex")

    print("\nDone.")

if __name__ == "__main__":
    main()
