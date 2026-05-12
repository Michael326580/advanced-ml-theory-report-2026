"""
复现《高级机器学习理论》课程报告中的核心实验表格与图表。

本脚本只读取 data/ 目录下已经审计确认的 CSV/JSON 结果，不依赖原始工业采集波形。
用途：
1. 复现报告中的数据集统计、主结果对比、模型消融和鲁棒性实验表；
2. 生成 accuracy-latency、fast student ablation 和 robustness summary 图；
3. 导出 JSON 与 Markdown 摘要，方便老师核查“报告结果是否由代码生成”。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

REQUIRED_INPUTS = {
    "dataset": DATA / "dataset_summary.csv",
    "main": DATA / "main_comparison.csv",
    "ablation": DATA / "fast_student_ablation.csv",
    "robustness": DATA / "robustness_results.csv",
    "calibration": DATA / "calibration.json",
}


def require_file(path: Path) -> None:
    """检查输入文件是否存在，缺失时给出明确报错。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少必要输入文件: {path}")


def read_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """读取全部审计数据表。"""
    for path in REQUIRED_INPUTS.values():
        require_file(path)
    dataset = pd.read_csv(REQUIRED_INPUTS["dataset"])
    main = pd.read_csv(REQUIRED_INPUTS["main"])
    ablation = pd.read_csv(REQUIRED_INPUTS["ablation"])
    robustness = pd.read_csv(REQUIRED_INPUTS["robustness"])
    calibration = json.loads(REQUIRED_INPUTS["calibration"].read_text(encoding="utf-8"))
    return dataset, main, ablation, robustness, calibration


def save_reproduced_tables(dataset: pd.DataFrame, main: pd.DataFrame,
                           ablation: pd.DataFrame, robustness: pd.DataFrame) -> None:
    """将输入审计表复制为 reproduced 版本，证明报告表格可由脚本复现。"""
    dataset.to_csv(OUT / "dataset_summary_reproduced.csv", index=False)
    main.to_csv(OUT / "main_comparison_reproduced.csv", index=False)
    ablation.to_csv(OUT / "fast_student_ablation_reproduced.csv", index=False)
    robustness.to_csv(OUT / "robustness_results_reproduced.csv", index=False)


def parse_latency_ms(text: str) -> float:
    """从 '0.8898 ms/sample' 等字符串中解析数值部分。"""
    return float(str(text).split()[0])


def plot_accuracy_latency(main: pd.DataFrame, ablation: pd.DataFrame) -> None:
    """绘制主模型与基线方法的精度-时延折中图。"""
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    compact = ablation.copy()
    ax.scatter(compact["Mean_ms"], compact["Pos_MAE_mm"], s=90, label="compact students")
    for _, row in compact.iterrows():
        ax.annotate(row["Model"].replace("plain_cnn_", ""),
                    (row["Mean_ms"], row["Pos_MAE_mm"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)

    refs = main[main["Method"].isin(["Teacher/PCCC reference", "Plain FFT table", "Attention student"])]
    for _, row in refs.iterrows():
        latency_value = parse_latency_ms(row["Mean_latency"])
        ax.scatter([latency_value], [row["MAE_mm"]], marker="x", s=85)
        ax.annotate(row["Method"], (latency_value, row["MAE_mm"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)

    ax.set_xlabel("Local timing (ms/sample or ms/segment core)")
    ax.set_ylabel("File-level position MAE (mm)")
    ax.set_title("Accuracy-latency trade-off")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_latency_tradeoff.png")
    plt.close(fig)


def plot_fast_student_ablation(ablation: pd.DataFrame) -> None:
    """绘制 w16/w24/w32 的消融实验图。"""
    fig, ax1 = plt.subplots(figsize=(7.6, 4.8), dpi=180)
    x = range(len(ablation))
    labels = [m.replace("plain_cnn_", "") for m in ablation["Model"]]
    ax1.bar(list(x), ablation["Pos_MAE_mm"], width=0.55, label="MAE (mm)")
    ax1.set_ylabel("Position MAE (mm)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Compact student width")
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(list(x), ablation["Mean_ms"], marker="o", label="Latency (ms)")
    ax2.set_ylabel("Mean latency (ms/sample)")
    ax1.set_title("Fast student ablation: accuracy and latency")
    fig.tight_layout()
    fig.savefig(OUT / "fast_student_ablation.png")
    plt.close(fig)


def plot_robustness(robustness: pd.DataFrame) -> None:
    """绘制关键扰动下 clean w24、robust w24 与 FFT proxy 的鲁棒性对比。"""
    selected = robustness[robustness["Condition"].isin([
        "Clean", "Gaussian 0.005", "Gaussian 0.01",
        "Channel shift 1 px", "Channel shift 2 px",
        "Amplitude scaling 0.02", "Spike noise 0.005",
    ])].copy()
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


def build_metrics_json(dataset: pd.DataFrame, main: pd.DataFrame,
                       ablation: pd.DataFrame, robustness: pd.DataFrame,
                       calibration: Dict) -> Dict:
    """汇总报告最核心指标，保存为机器可读 JSON。"""
    w24 = ablation[ablation["Model"] == "plain_cnn_w24"].iloc[0].to_dict()
    robust_gain = robustness[robustness["Condition"].isin(["Gaussian 0.005", "Gaussian 0.01", "Channel shift 1 px"])]
    return {
        "selected_model": "plain_cnn_w24",
        "w24_pos_mae_mm": float(w24["Pos_MAE_mm"]),
        "w24_rmse_mm": float(w24["RMSE_mm"]),
        "w24_latency_ms": float(w24["Mean_ms"]),
        "w24_params": int(w24["Params"]),
        "w24_macs_million": float(w24["MACs_M"]),
        "calibration": calibration,
        "robustness_key_cases": robust_gain.to_dict(orient="records"),
        "dataset_items": dataset.to_dict(orient="records"),
        "boundary_note": "RMSE is validation predictive error, not full GUM uncertainty; synthetic demo is not real industrial validation."
    }


def write_summary_md(metrics: Dict, main: pd.DataFrame, ablation: pd.DataFrame, robustness: pd.DataFrame) -> None:
    """生成 Markdown 摘要，便于老师快速核查。"""
    lines = [
        "# 课程报告核心结果复现摘要",
        "",
        "## 1. 选定模型",
        f"- 模型：{metrics['selected_model']}",
        f"- MAE：{metrics['w24_pos_mae_mm']:.6f} mm",
        f"- RMSE：{metrics['w24_rmse_mm']:.6f} mm",
        f"- 平均延迟：{metrics['w24_latency_ms']:.6f} ms/sample",
        f"- 参数量：{metrics['w24_params']:,}",
        f"- MACs：{metrics['w24_macs_million']:.3f} M",
        "",
        "## 2. 主结果表",
        main.to_markdown(index=False),
        "",
        "## 3. Fast student 消融",
        ablation.to_markdown(index=False),
        "",
        "## 4. 鲁棒性结果",
        robustness.to_markdown(index=False),
        "",
        "## 5. 结论边界",
        "- Teacher/PCCC 是确定性教师参考，不是独立真实值。",
        "- plain_cnn_w24 的时延是本地 CPU single-thread 协议下的 PyTorch timing。",
        "- robust_w24 主要改善 Gaussian noise 和 small channel shift，不能宣称全扰动优于 FFT proxy。",
    ]
    (OUT / "reproduced_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    dataset, main_tbl, ablation, robustness, calibration = read_tables()
    save_reproduced_tables(dataset, main_tbl, ablation, robustness)
    plot_accuracy_latency(main_tbl, ablation)
    plot_fast_student_ablation(ablation)
    plot_robustness(robustness)

    metrics = build_metrics_json(dataset, main_tbl, ablation, robustness, calibration)
    (OUT / "reproduced_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(metrics, main_tbl, ablation, robustness)

    print("Dataset summary:")
    print(dataset.to_string(index=False))
    print("\nMain comparison:")
    print(main_tbl.to_string(index=False))
    print("\nFast student ablation:")
    print(ablation.to_string(index=False))
    print("\nRobustness results:")
    print(robustness.to_string(index=False))
    print(f"\n复现完成，输出文件已保存到: {OUT}")


if __name__ == "__main__":
    main()
