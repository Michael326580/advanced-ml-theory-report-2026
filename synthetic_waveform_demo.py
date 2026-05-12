"""
合成双通道波形算法演示。

真实锥光全息项目的原始工业采集波形不随课程包公开。本脚本生成可控的双通道反相
干涉条纹，用于演示课程报告中的算法流程：

合成 2×2048 双通道 waveform → 简化确定性 FFT teacher 估计 W(pixel)
→ 有理标定模型换算 x(mm) → 可选轻量 CNN 回归演示。

注意：本脚本是教学演示和流程验证，不替代 data/*.csv 中的真实项目审计结果。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# 真实项目使用的单段长度和有理标定参数。
N = 2048
CAL_A = 2485.85
CAL_B = 0.489856
CAL_C = 92.7040


def width_to_displacement(width: np.ndarray | float) -> np.ndarray:
    """将条纹宽度 W(pixel) 映射为重构位移 x(mm)。"""
    return CAL_A / (np.asarray(width) - CAL_B) + CAL_C


def make_dual_channel(width: float, noise: float = 0.02,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """生成一条合成双通道反相干涉波形。"""
    if rng is None:
        rng = np.random.default_rng(0)
    n = np.arange(N, dtype=np.float32)
    phase = rng.uniform(0, 2 * np.pi)
    freq = 1.0 / (2.0 * width)  # 与教师定义 W=1/(2f) 保持一致。
    baseline = 0.08 * np.sin(2 * np.pi * n / N)
    ch1 = np.sin(2 * np.pi * freq * n + phase) + baseline
    ch2 = -np.sin(2 * np.pi * freq * n + phase + rng.normal(0, 0.02)) + 0.8 * baseline
    ch1 += rng.normal(0, noise, size=N)
    ch2 += rng.normal(0, noise, size=N)
    return np.stack([ch1, ch2]).astype(np.float32)


def teacher_fft_width(sample: np.ndarray) -> float:
    """简化确定性 teacher：差分信号 + 加窗 FFT + 三点抛物线峰值细化。"""
    differential = sample[0] - sample[1]
    differential = differential - differential.mean()
    window = np.hanning(len(differential))
    spectrum = np.abs(np.fft.rfft(differential * window))
    spectrum[:2] = 0.0
    k0 = int(np.argmax(spectrum))
    if 1 <= k0 < len(spectrum) - 1:
        denom = spectrum[k0 - 1] - 2 * spectrum[k0] + spectrum[k0 + 1]
        delta = 0.5 * (spectrum[k0 - 1] - spectrum[k0 + 1]) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    f_hat = (k0 + delta) / len(differential)
    return float(1.0 / (2.0 * f_hat))


def generate_dataset(n_samples: int, seed: int, noise: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成合成数据集，并用 teacher 得到估计宽度。"""
    rng = np.random.default_rng(seed)
    widths = rng.uniform(18.6125, 28.9355, size=n_samples).astype(np.float32)
    samples = np.stack([make_dual_channel(float(w), noise=noise, rng=rng) for w in widths])
    teacher_w = np.array([teacher_fft_width(s) for s in samples], dtype=np.float32)
    return samples, widths, teacher_w


def plot_samples(samples: np.ndarray, widths: np.ndarray, teacher_w: np.ndarray) -> None:
    """绘制合成双通道波形和 teacher 估计结果。"""
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    idx = 0
    ax.plot(samples[idx, 0, :300], label="Channel 1")
    ax.plot(samples[idx, 1, :300], label="Channel 2")
    ax.set_title(f"Synthetic waveform: true W={widths[idx]:.3f} px, teacher W={teacher_w[idx]:.3f} px")
    ax.set_xlabel("Pixel index")
    ax.set_ylabel("Normalized intensity")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_waveforms.png")
    fig.savefig(OUT / "synthetic_waveform_demo.png")
    plt.close(fig)


def run_torch_demo(samples: np.ndarray, labels: np.ndarray, epochs: int, seed: int) -> Dict[str, float]:
    """可选 CNN 演示：若未安装 PyTorch，则自动跳过。"""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover
        print(f"PyTorch 未安装或不可用，跳过 CNN 演示: {exc}")
        return {"cnn_demo_available": 0.0}

    torch.manual_seed(seed)
    split = int(len(samples) * 0.8)
    x_train = torch.tensor(samples[:split], dtype=torch.float32)
    y_train_raw = labels[:split]
    x_val = torch.tensor(samples[split:], dtype=torch.float32)
    y_val_raw = labels[split:]
    y_min = float(labels.min())
    y_range = float(labels.max() - labels.min())
    y_train = torch.tensor((y_train_raw - y_min) / y_range, dtype=torch.float32).unsqueeze(1)

    class PlainCNN(nn.Module):
        """教学版 compact 1D-CNN；结构对应报告中的 plain_cnn_w24 思路。"""
        def __init__(self, width: int = 24):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(2, width, kernel_size=7, padding=3), nn.ReLU(),
                nn.Conv1d(width, width, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(width, width, kernel_size=5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = PlainCNN(24)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_norm = model(x_val).squeeze(1).numpy()
    pred_w = pred_norm * y_range + y_min
    mae_w = float(np.mean(np.abs(pred_w - y_val_raw)))
    mae_x = float(np.mean(np.abs(width_to_displacement(pred_w) - width_to_displacement(y_val_raw))))
    return {"cnn_demo_available": 1.0, "cnn_val_width_mae_px": mae_w, "cnn_val_position_mae_mm": mae_x}


def main() -> None:
    parser = argparse.ArgumentParser(description="合成双通道条纹与 FFT teacher/CNN 演示")
    parser.add_argument("--n-samples", type=int, default=120, help="合成样本数")
    parser.add_argument("--epochs", type=int, default=5, help="可选 CNN 演示训练轮数")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--noise", type=float, default=0.02, help="合成高斯噪声强度")
    args = parser.parse_args()

    samples, true_w, teacher_w = generate_dataset(args.n_samples, args.seed, args.noise)
    plot_samples(samples, true_w, teacher_w)
    metrics: Dict[str, float] = {
        "n_samples": float(args.n_samples),
        "noise": float(args.noise),
        "teacher_width_mae_px": float(np.mean(np.abs(teacher_w - true_w))),
        "teacher_position_mae_mm": float(np.mean(np.abs(width_to_displacement(teacher_w) - width_to_displacement(true_w)))),
    }
    metrics.update(run_torch_demo(samples, teacher_w, args.epochs, args.seed))
    pd.DataFrame([metrics]).to_csv(OUT / "synthetic_demo_metrics.csv", index=False)
    (OUT / "synthetic_demo_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"合成波形演示完成，输出文件保存到: {OUT}")


if __name__ == "__main__":
    main()
