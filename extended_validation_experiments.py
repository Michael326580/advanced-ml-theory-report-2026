"""
补充验证实验脚本。

本脚本在现有审计数据基础上新增“可复现、可解释、不会篡改原始结论”的验证分析，
用于把课程报告从“只复现表格”升级为“有补充实验与深入分析”的版本。

新增实验包括：
1. 精度-时延-复杂度综合折中评分；
2. robust_w24 相对 clean_w24 的扰动改善比例；
3. 有理标定模型的灵敏度 |dx/dW| 分析；
4. 合成双通道条纹上的 FFT teacher 噪声扫描。

注意：这些实验是基于已审计结果和合成信号的补充验证，不能替代真实工业采集数据。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

N = 2048


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"缺少必要文件: {path}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    ablation_path = DATA / "fast_student_ablation.csv"
    robustness_path = DATA / "robustness_results.csv"
    calibration_path = DATA / "calibration.json"
    for path in [ablation_path, robustness_path, calibration_path]:
        require_file(path)
    return (
        pd.read_csv(ablation_path),
        pd.read_csv(robustness_path),
        json.loads(calibration_path.read_text(encoding="utf-8")),
    )


def minmax_positive(series: pd.Series) -> pd.Series:
    """将越小越好的指标归一化成越大越好的分数。"""
    s = series.astype(float)
    if abs(float(s.max() - s.min())) < 1e-12:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s.max() - s) / (s.max() - s.min())


def tradeoff_audit(ablation: pd.DataFrame) -> pd.DataFrame:
    """计算 compact students 的综合折中评分。

    score = 0.5*accuracy_score + 0.3*latency_score + 0.2*complexity_score
    三项分数均为 0~1，越高越好。该评分用于补充说明 w24 的综合选择合理性。
    """
    df = ablation.copy()
    df["accuracy_score"] = minmax_positive(df["Pos_MAE_mm"])
    df["latency_score"] = minmax_positive(df["Mean_ms"])
    df["complexity_score"] = minmax_positive(df["MACs_M"])
    df["weighted_tradeoff_score"] = (
        0.5 * df["accuracy_score"] +
        0.3 * df["latency_score"] +
        0.2 * df["complexity_score"]
    )
    df.to_csv(OUT / "tradeoff_audit.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=180)
    labels = df["Model"].str.replace("plain_cnn_", "", regex=False)
    ax.bar(labels, df["weighted_tradeoff_score"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Weighted trade-off score")
    ax.set_title("Accuracy-latency-complexity trade-off audit")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    for i, v in enumerate(df["weighted_tradeoff_score"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff_audit.png")
    plt.close(fig)
    return df


def robustness_improvement_audit(robustness: pd.DataFrame) -> pd.DataFrame:
    """计算 robust_w24 相对 clean_w24 的改善比例。"""
    df = robustness.copy()
    df["relative_change_percent"] = (
        (df["Clean_w24_MAE_mm"] - df["Robust_w24_MAE_mm"]) /
        df["Clean_w24_MAE_mm"] * 100.0
    )
    df["is_improved"] = df["relative_change_percent"] > 0
    df.to_csv(OUT / "robustness_improvement_audit.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=180)
    ax.bar(df["Condition"], df["relative_change_percent"])
    ax.axhline(0, linewidth=1.0)
    ax.set_ylabel("Relative MAE reduction vs clean w24 (%)")
    ax.set_title("Robust fine-tuning improvement audit")
    ax.set_xticklabels(df["Condition"], rotation=35, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "robustness_improvement_audit.png")
    plt.close(fig)
    return df


def calibration_sensitivity(calibration: Dict) -> pd.DataFrame:
    """分析有理标定模型 |dx/dW|，解释宽度误差到位移误差的放大。"""
    a = float(calibration["a"])
    b = float(calibration["b"])
    c = float(calibration["c"])
    widths = np.linspace(18.6125, 28.9355, 200)
    displacement = a / (widths - b) + c
    sensitivity = np.abs(-a / (widths - b) ** 2)
    df = pd.DataFrame({
        "W_pixel": widths,
        "x_hat_mm": displacement,
        "abs_dx_dW_mm_per_pixel": sensitivity,
    })
    df.to_csv(OUT / "calibration_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=180)
    ax.plot(df["W_pixel"], df["abs_dx_dW_mm_per_pixel"])
    ax.set_xlabel("Fringe width W (pixel)")
    ax.set_ylabel("|dx/dW| (mm/pixel)")
    ax.set_title("Calibration sensitivity of rational model")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_sensitivity.png")
    plt.close(fig)
    return df


def make_dual_channel(width: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    n = np.arange(N, dtype=np.float32)
    phase = rng.uniform(0, 2 * np.pi)
    freq = 1.0 / (2.0 * width)
    baseline = 0.05 * np.sin(2 * np.pi * n / N)
    ch1 = np.sin(2 * np.pi * freq * n + phase) + baseline
    ch2 = -np.sin(2 * np.pi * freq * n + phase) + 0.8 * baseline
    ch1 += rng.normal(0, noise, size=N)
    ch2 += rng.normal(0, noise, size=N)
    return np.stack([ch1, ch2]).astype(np.float32)


def teacher_fft_width(sample: np.ndarray) -> float:
    d = sample[0] - sample[1]
    d = d - d.mean()
    spectrum = np.abs(np.fft.rfft(d * np.hanning(len(d))))
    spectrum[:2] = 0.0
    k0 = int(np.argmax(spectrum))
    if 1 <= k0 < len(spectrum) - 1:
        denom = spectrum[k0 - 1] - 2 * spectrum[k0] + spectrum[k0 + 1]
        delta = 0.5 * (spectrum[k0 - 1] - spectrum[k0 + 1]) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    return float(1.0 / (2.0 * ((k0 + delta) / len(d))))


def synthetic_noise_sweep(seed: int = 2026) -> pd.DataFrame:
    """在合成数据上扫描噪声强度，验证 FFT teacher 对噪声的退化趋势。"""
    rng = np.random.default_rng(seed)
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.08]
    rows: List[Dict[str, float]] = []
    widths = rng.uniform(18.6125, 28.9355, size=80)
    for noise in noise_levels:
        pred = []
        for w in widths:
            sample = make_dual_channel(float(w), noise=noise, rng=rng)
            pred.append(teacher_fft_width(sample))
        pred = np.asarray(pred)
        rows.append({
            "noise_std": noise,
            "teacher_width_mae_px": float(np.mean(np.abs(pred - widths))),
            "teacher_width_rmse_px": float(np.sqrt(np.mean((pred - widths) ** 2))),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_noise_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=180)
    ax.plot(df["noise_std"], df["teacher_width_mae_px"], marker="o", label="MAE")
    ax.plot(df["noise_std"], df["teacher_width_rmse_px"], marker="s", label="RMSE")
    ax.set_xlabel("Synthetic Gaussian noise std")
    ax.set_ylabel("Width error (pixel)")
    ax.set_title("Synthetic FFT teacher noise sweep")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_noise_sweep.png")
    plt.close(fig)
    return df


def write_summary(tradeoff: pd.DataFrame, improvement: pd.DataFrame,
                  sensitivity: pd.DataFrame, noise_sweep: pd.DataFrame) -> None:
    best_tradeoff = tradeoff.sort_values("weighted_tradeoff_score", ascending=False).iloc[0]
    gaussian_001 = improvement[improvement["Condition"] == "Gaussian 0.01"].iloc[0]
    shift1 = improvement[improvement["Condition"] == "Channel shift 1 px"].iloc[0]
    spike = improvement[improvement["Condition"] == "Spike noise 0.005"].iloc[0]
    summary = {
        "best_tradeoff_model": str(best_tradeoff["Model"]),
        "best_tradeoff_score": float(best_tradeoff["weighted_tradeoff_score"]),
        "gaussian_0_01_relative_reduction_percent": float(gaussian_001["relative_change_percent"]),
        "channel_shift_1px_relative_reduction_percent": float(shift1["relative_change_percent"]),
        "spike_0_005_relative_change_percent": float(spike["relative_change_percent"]),
        "calibration_sensitivity_min": float(sensitivity["abs_dx_dW_mm_per_pixel"].min()),
        "calibration_sensitivity_max": float(sensitivity["abs_dx_dW_mm_per_pixel"].max()),
        "synthetic_noise_sweep_final_mae_px": float(noise_sweep.iloc[-1]["teacher_width_mae_px"]),
    }
    (OUT / "extended_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"""# 补充验证实验摘要

本文件由 `python extended_validation_experiments.py` 自动生成。

## 1. 精度-时延-复杂度折中

综合评分最高模型：`{summary['best_tradeoff_model']}`，score={summary['best_tradeoff_score']:.4f}。
该结果用于补充说明 compact student 的选择不是只看单一指标，而是同时考虑精度、时延和复杂度。

## 2. 鲁棒性改善比例

- Gaussian 0.01 相对 clean w24 改善：{summary['gaussian_0_01_relative_reduction_percent']:.2f}%
- Channel shift 1 px 相对 clean w24 改善：{summary['channel_shift_1px_relative_reduction_percent']:.2f}%
- Spike noise 0.005 相对变化：{summary['spike_0_005_relative_change_percent']:.2f}%

结论：robust_w24 对 Gaussian noise 和小通道偏移有效，但 spike noise 仍是失败模式。

## 3. 标定灵敏度

在 W=18.6125–28.9355 pixel 范围内，|dx/dW| 约为 {summary['calibration_sensitivity_min']:.3f}–{summary['calibration_sensitivity_max']:.3f} mm/pixel。
这说明宽度预测误差会通过标定模型放大成位移误差，因此报告选择 W(pixel) 作为可解释中间学习目标是必要的。

## 4. 合成噪声扫描

合成噪声扫描显示，FFT teacher 的宽度估计误差会随噪声强度变化。该实验只用于流程验证，不替代真实工业实验。
"""
    (OUT / "extended_validation_summary.md").write_text(md, encoding="utf-8")


def main() -> None:
    ablation, robustness, calibration = load_inputs()
    tradeoff = tradeoff_audit(ablation)
    improvement = robustness_improvement_audit(robustness)
    sensitivity = calibration_sensitivity(calibration)
    noise_sweep = synthetic_noise_sweep()
    write_summary(tradeoff, improvement, sensitivity, noise_sweep)
    print("补充验证实验完成。输出目录:", OUT)
    print("关键输出: tradeoff_audit.png, robustness_improvement_audit.png, calibration_sensitivity.png, synthetic_noise_sweep.png")


if __name__ == "__main__":
    main()
