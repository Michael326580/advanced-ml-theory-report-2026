"""Reproduce core tables and figures for the course report.

This script reads audited result tables saved in data/*.csv and regenerates
summary CSV files and two figures used by the report. It does not require the
private raw acquisition waveforms.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def read_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(DATA / "dataset_summary.csv")
    main = pd.read_csv(DATA / "main_comparison.csv")
    ablation = pd.read_csv(DATA / "fast_student_ablation.csv")
    robustness = pd.read_csv(DATA / "robustness_results.csv")
    return dataset, main, ablation, robustness


def save_reproduced_tables(dataset: pd.DataFrame, main: pd.DataFrame,
                           ablation: pd.DataFrame, robustness: pd.DataFrame) -> None:
    dataset.to_csv(OUT / "dataset_summary_reproduced.csv", index=False)
    main.to_csv(OUT / "main_comparison_reproduced.csv", index=False)
    ablation.to_csv(OUT / "fast_student_ablation_reproduced.csv", index=False)
    robustness.to_csv(OUT / "robustness_results_reproduced.csv", index=False)


def plot_accuracy_latency(main: pd.DataFrame, ablation: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)

    compact = ablation.copy()
    ax.scatter(compact["Mean_ms"], compact["Pos_MAE_mm"], s=80, label="compact students")
    for _, row in compact.iterrows():
        ax.annotate(row["Model"].replace("plain_cnn_", ""),
                    (row["Mean_ms"], row["Pos_MAE_mm"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)

    refs = main[main["Method"].isin(["Teacher/PCCC reference", "Plain FFT table", "Attention student"])]
    for _, row in refs.iterrows():
        latency_value = float(str(row["Mean_latency"]).split()[0])
        ax.scatter([latency_value], [row["MAE_mm"]], marker="x", s=75)
        ax.annotate(row["Method"], (latency_value, row["MAE_mm"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)

    ax.set_xlabel("Local timing (ms/sample or ms/segment core)")
    ax.set_ylabel("File-level position MAE (mm)")
    ax.set_title("Accuracy-latency trade-off")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_latency_tradeoff.png")
    plt.close(fig)


def plot_robustness(robustness: pd.DataFrame) -> None:
    selected = robustness[robustness["Condition"].isin([
        "Clean", "Gaussian 0.005", "Gaussian 0.01",
        "Channel shift 1 px", "Channel shift 2 px",
        "Amplitude scaling 0.02", "Spike noise 0.005",
    ])].copy()
    x = range(len(selected))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
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


def main() -> None:
    dataset, main_tbl, ablation, robustness = read_tables()
    save_reproduced_tables(dataset, main_tbl, ablation, robustness)
    plot_accuracy_latency(main_tbl, ablation)
    plot_robustness(robustness)

    print("Dataset summary:")
    print(dataset.to_string(index=False))
    print("\nMain comparison:")
    print(main_tbl.to_string(index=False))
    print("\nFast student ablation:")
    print(ablation.to_string(index=False))
    print("\nRobustness results:")
    print(robustness.to_string(index=False))
    print(f"\nReproduced outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
