"""复现课程报告核心表格与图表。

本脚本只读取仓库 data/ 目录下随包提供的审计结果表，不读取原始工业采集波形。
用途：
1. 复现课程报告中的数据集表、主结果对比表、compact student 消融表和鲁棒性表；
2. 生成 accuracy-latency、fast student ablation、robustness summary 三类图；
3. 导出 JSON 与 Markdown 摘要，便于课程老师快速核验报告结果来源。

注意：本脚本复现的是完整 deep-study 项目最终审计后的指标，而不是重新训练真实模型。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# 仓库根目录、输入数据目录和输出目录。
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


REQUIRED_INPUTS = {
    "dataset": DATA / "dataset_summary.csv",
    "main": DATA / "main_comparison.csv",
    "ablation": DATA / "fast_student_ablation.csv",
    "robustness": DATA / "robustness_results.csv",
}


def require_file(path: Path) -> None:
    """检查输入文件是否存在。

    参数：
        path: 需要检查的文件路径。

    抛出：
        FileNotFoundError: 当文件不存在时给出清晰错误，避免静默失败。
    """
    if not path.exists():
        raise FileNotFoundError(f"缺少必要输入文件: {path}")


def read_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取课程报告使用的四类审计结果表。

    返回：
        dataset: 数据集规模和范围表；
        main: 主方法精度与延迟对比表；
        ablation: w16/w24/w32 compact student 消融表；
        robustness: 扰动鲁棒性结果表。
    """
    for path in REQUIRED_INPUTS.values():
        require_file(path)

    dataset = pd.read_csv(REQUIRED_INPUTS["dataset"])
    main = pd.read_csv(REQUIRED_INPUTS["main"])
    ablation = pd.read_csv(REQUIRED_INPUTS["ablation"])
    robustness = pd.read_csv(REQUIRED_INPUTS["robustness"])
    return dataset, main, ablation, robustness


def save_reproduced_tables(
    dataset: pd.DataFrame,
    main: pd.DataFrame,
    ablation: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    """将读取到的审计表复制导出到 outputs/，证明报告数据可复核。"""
    dataset.to_csv(OUT / "dataset_summary_reproduced.csv", index=False)
    main.to_csv(OUT / "main_comparison_reproduced.csv", index=False)
    ablation.to_csv(OUT / "fast_student_ablation_reproduced.csv", index=False)
    robustness.to_csv(OUT / "robustness_results_reproduced.csv", index=False)


def parse_latency_ms(text: str) -> float:
    """从 '0.8898 ms/sample' 等字符串中解析数值部分。"""
    return float(str(text).split()[0])


def plot_accuracy_latency(main: pd.DataFrame, ablation: pd.DataFrame) -> None:
    """绘制精度-时延折中图。

    图中 compact students 来自 w16/w24/w32 消融表；Teacher、Plain FFT 和 Attention
    student 作为参考点。横轴单位混合 ms/sample 与 ms/segment core，因此图题和报告中
    必须明确该图只用于本地协议下的 timing 对比，不能夸大为完整端到端速度结论。
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)

    compact = ablation.copy()
    ax.scatter(compact["Mean_ms"], compact["Pos_MAE_mm"], s=90, label="compact students")
    for _, row in compact.iterrows():
        ax.annotate(
            row["Model"].replace("plain_cnn_", ""),
            (row["Mean_ms"], row["Pos_MAE_mm"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    refs = main[main["Method"].isin(["Teacher/PCCC reference", "Plain FFT table", "Attention student"])]
    for _, row in refs.iterrows():
        latency_value = parse_latency_ms(row["Mean_latency"])
        ax.scatter([latency_value], [row["MAE_mm"]], marker="x", s=85)
        ax.annotate(
            row["Method"],
            (latency_value, row["MAE_mm"]),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=8,
        )

    ax.set_xlabel("Local timing (ms/sample or ms/segment core)")
    ax.set_ylabel("File-level position MAE (mm)")
    ax.set_title("Accuracy-latency trade-off")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_latency_tradeoff.png")
    plt.close(fig)


def plot_fast_student_ablation(ablation: pd.DataFrame) -> None:
    """绘制 w16/w24/w32 的精度、时延与复杂度综合图。"""
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    x = range(len(ablation))
    labels = [m.replace("plain_cnn_", "") for m in ablation["Model"]]

    ax1.bar(list(x), ablation["Pos_MAE_mm"], width=0.42, label="MAE (mm)")
    ax1.set_ylabel("Position MAE (mm)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Compact student width")

    ax2 = ax1.twinx()
    ax2.plot(list(x), ablation["Mean_ms"], marker="o", label="Latency (ms)")
    ax2.set_ylabel("Mean latency (ms/sample)")

    for i, row in ablation.iterrows():
        ax1.text(i, row["Pos_MAE_mm"] + 0.003, f"{row['Pos_MAE_mm']:.4f}", ha="center", fontsize=8)
        ax2.annotate(f"{row['Params']:,} params", (i, row["Mean_ms"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)

    ax1.set_title("Fast compact-student ablation")
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fast_student_ablation.png")
    plt.close(fig)


def plot_robustness(robustness: pd.DataFrame) -> None:
    """绘制鲁棒性摘要图，突出改善项和失败项。"""
    selected = robustness[
        robustness["Condition"].isin(
            [
                "Clean",
                "Gaussian 0.005",
                "Gaussian 0.01",
                "Channel shift 1 px",
                "Channel shift 2 px",
                "Amplitude scaling 0.02",
                "Spike noise 0.005",
            ]
        )
    ].copy()
    x = range(len(selected))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10.8, 5.3), dpi=180)
    ax.bar([i - width for i in x], selected["Clean_w24_MAE_mm"], width=width, label="clean w24")
    ax.bar(list(x), selected["Robust_w24_MAE_mm"], width=width, label="robust w24")
    ax.bar([i + width for i in x], selected["FFT_proxy_MAE_mm"], width=width, label="FFT proxy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(selected["Condition"], rotation=25, ha="right")
    ax.set_ylabel("File-level position MAE (mm)")
    ax.set_title("Robustness summary under waveform perturbations")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "robustness_summary.png")
    plt.close(fig)


def build_metrics(main: pd.DataFrame, ablation: pd.DataFrame, robustness: pd.DataFrame) -> Dict[str, float | str]:
    """从审计表中提取课程报告最关键的结论指标。"""
    clean_w24 = main[main["Method"] == "Clean plain_cnn_w24"].iloc[0]
    plain_fft = main[main["Method"] == "Plain FFT table"].iloc[0]
    teacher = main[main["Method"] == "Teacher/PCCC reference"].iloc[0]
    best_ablation = ablation.sort_values("Pos_MAE_mm").iloc[0]

    g001 = robustness[robustness["Condition"] == "Gaussian 0.01"].iloc[0]
    shift1 = robustness[robustness["Condition"] == "Channel shift 1 px"].iloc[0]
    spike = robustness[robustness["Condition"] == "Spike noise 0.005"].iloc[0]

    return {
        "clean_w24_mae_mm": float(clean_w24["MAE_mm"]),
        "clean_w24_rmse_mm": float(clean_w24["RMSE_mm"]),
        "clean_w24_latency": str(clean_w24["Mean_latency"]),
        "plain_fft_mae_mm": float(plain_fft["MAE_mm"]),
        "teacher_mae_mm": float(teacher["MAE_mm"]),
        "best_ablation_by_mae": str(best_ablation["Model"]),
        "gaussian_001_clean_w24_mae": float(g001["Clean_w24_MAE_mm"]),
        "gaussian_001_robust_w24_mae": float(g001["Robust_w24_MAE_mm"]),
        "channel_shift_1_clean_w24_mae": float(shift1["Clean_w24_MAE_mm"]),
        "channel_shift_1_robust_w24_mae": float(shift1["Robust_w24_MAE_mm"]),
        "spike_0005_clean_w24_mae": float(spike["Clean_w24_MAE_mm"]),
        "spike_0005_robust_w24_mae": float(spike["Robust_w24_MAE_mm"]),
    }


def write_summary_md(metrics: Dict[str, float | str]) -> None:
    """导出 Markdown 摘要，作为报告实验结果的可核验说明。"""
    lines = [
        "# 课程报告核心结果复现摘要",
        "",
        "本文件由 `python reproduce_report_results.py` 自动生成。",
        "",
        "## 主模型结果",
        f"- clean plain_cnn_w24: MAE={metrics['clean_w24_mae_mm']:.4f} mm, RMSE={metrics['clean_w24_rmse_mm']:.4f} mm, latency={metrics['clean_w24_latency']}。",
        f"- Plain FFT baseline: MAE={metrics['plain_fft_mae_mm']:.4f} mm。",
        f"- Teacher/PCCC reference: MAE={metrics['teacher_mae_mm']:.4f} mm。",
        "",
        "## 消融结论",
        f"- 在 w16/w24/w32 中，按 MAE 排序的最优模型为 `{metrics['best_ablation_by_mae']}`。课程报告选择 w24 作为综合精度-延迟折中模型。",
        "",
        "## 鲁棒性结论",
        f"- Gaussian 0.01: clean w24 MAE={metrics['gaussian_001_clean_w24_mae']:.4f} mm, robust w24 MAE={metrics['gaussian_001_robust_w24_mae']:.4f} mm。",
        f"- Channel shift 1 px: clean w24 MAE={metrics['channel_shift_1_clean_w24_mae']:.4f} mm, robust w24 MAE={metrics['channel_shift_1_robust_w24_mae']:.4f} mm。",
        f"- Spike noise 0.005: clean w24 MAE={metrics['spike_0005_clean_w24_mae']:.4f} mm, robust w24 MAE={metrics['spike_0005_robust_w24_mae']:.4f} mm，说明 spike noise 仍是失败模式。",
        "",
        "## 结论边界",
        "- Teacher/PCCC 是确定性教师参考和伪标签生成流程，不是独立真实值。",
        "- Local CPU timing 不能等同完整端到端系统速度。",
        "- robust_w24 不能被表述为在所有扰动下优于 FFT proxy。",
    ]
    (OUT / "reproduced_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """主入口：读取、复现、绘图、导出摘要。"""
    dataset, main_tbl, ablation, robustness = read_tables()
    save_reproduced_tables(dataset, main_tbl, ablation, robustness)
    plot_accuracy_latency(main_tbl, ablation)
    plot_fast_student_ablation(ablation)
    plot_robustness(robustness)

    metrics = build_metrics(main_tbl, ablation, robustness)
    (OUT / "reproduced_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(metrics)

    print("Dataset summary:")
    print(dataset.to_string(index=False))
    print("\nMain comparison:")
    print(main_tbl.to_string(index=False))
    print("\nFast student ablation:")
    print(ablation.to_string(index=False))
    print("\nRobustness results:")
    print(robustness.to_string(index=False))
    print(f"\n复现完成，输出文件保存到: {OUT}")
    print("关键输出: accuracy_latency_tradeoff.png, fast_student_ablation.png, robustness_summary.png, reproduced_metrics.json")


if __name__ == "__main__":
    main()
