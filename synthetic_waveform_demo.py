"""Synthetic demonstration of deterministic-teacher width estimation and CNN regression.

The real project uses private dual-channel line-array acquisition records. This
script generates synthetic anti-phase fringe waveforms to demonstrate the same
algorithmic structure without exposing raw industrial data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

N = 2048
CAL_A = 2485.85
CAL_B = 0.489856
CAL_C = 92.7040


def width_to_displacement(width: np.ndarray) -> np.ndarray:
    return CAL_A / (width - CAL_B) + CAL_C


def make_dual_channel(width: float, noise: float = 0.02, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    n = np.arange(N, dtype=np.float32)
    phase = rng.uniform(0, 2 * np.pi)
    # The teacher definition uses W = 1/(2f), so the sinusoid frequency is f = 1/(2W).
    f = 1.0 / (2.0 * width)
    base = 0.08 * np.sin(2 * np.pi * n / N)  # slow baseline drift
    ch1 = np.sin(2 * np.pi * f * n + phase) + base
    ch2 = -np.sin(2 * np.pi * f * n + phase + rng.normal(0, 0.02)) + 0.8 * base
    ch1 += rng.normal(0, noise, size=N)
    ch2 += rng.normal(0, noise, size=N)
    return np.stack([ch1, ch2]).astype(np.float32)


def teacher_fft_width(sample: np.ndarray) -> float:
    # simplified deterministic teacher: remove mean, use anti-phase differential signal, FFT peak + parabolic interpolation
    d = sample[0] - sample[1]
    d = d - d.mean()
    window = np.hanning(len(d))
    spec = np.abs(np.fft.rfft(d * window))
    spec[:2] = 0.0
    k0 = int(np.argmax(spec))
    if 1 <= k0 < len(spec) - 1:
        denom = spec[k0 - 1] - 2 * spec[k0] + spec[k0 + 1]
        delta = 0.5 * (spec[k0 - 1] - spec[k0 + 1]) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    f_hat = (k0 + delta) / len(d)
    return float(1.0 / (2.0 * f_hat))


def generate_dataset(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    widths = rng.uniform(18.6125, 28.9355, size=n_samples).astype(np.float32)
    samples = np.stack([make_dual_channel(float(w), rng=rng) for w in widths])
    teacher_w = np.array([teacher_fft_width(s) for s in samples], dtype=np.float32)
    return samples, widths, teacher_w


def plot_samples(samples: np.ndarray, widths: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    idx = 0
    ax.plot(samples[idx, 0, :300], label="Channel 1")
    ax.plot(samples[idx, 1, :300], label="Channel 2")
    ax.set_title(f"Synthetic anti-phase dual-channel waveform, true W={widths[idx]:.3f} px")
    ax.set_xlabel("Pixel index")
    ax.set_ylabel("Normalized intensity")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_waveforms.png")
    plt.close(fig)


def run_torch_demo(samples: np.ndarray, labels: np.ndarray, epochs: int, seed: int) -> dict[str, float]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover
        print(f"PyTorch is unavailable, skipping CNN demo: {exc}")
        return {"cnn_demo_available": 0.0}

    torch.manual_seed(seed)
    n = len(samples)
    split = int(n * 0.8)
    x_train = torch.tensor(samples[:split], dtype=torch.float32)
    y_train_raw = labels[:split]
    x_val = torch.tensor(samples[split:], dtype=torch.float32)
    y_val_raw = labels[split:]
    y_min = float(labels.min())
    y_range = float(labels.max() - labels.min())
    y_train = torch.tensor((y_train_raw - y_min) / y_range, dtype=torch.float32).unsqueeze(1)

    class PlainCNN(nn.Module):
        def __init__(self, width: int = 24):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(2, width, kernel_size=7, padding=3), nn.ReLU(),
                nn.Conv1d(width, width, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(width, width, kernel_size=5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, 1)
            )
        def forward(self, x):
            return self.net(x)

    model = PlainCNN(24)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_norm = model(x_val).squeeze(1).numpy()
    pred_w = pred_norm * y_range + y_min
    mae_w = float(np.mean(np.abs(pred_w - y_val_raw)))
    pred_x = width_to_displacement(pred_w)
    true_x = width_to_displacement(y_val_raw)
    mae_x = float(np.mean(np.abs(pred_x - true_x)))
    return {"cnn_demo_available": 1.0, "cnn_val_width_mae_px": mae_w, "cnn_val_position_mae_mm": mae_x}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    samples, true_w, teacher_w = generate_dataset(args.n_samples, args.seed)
    plot_samples(samples, true_w)
    teacher_width_mae = float(np.mean(np.abs(teacher_w - true_w)))
    teacher_pos_mae = float(np.mean(np.abs(width_to_displacement(teacher_w) - width_to_displacement(true_w))))
    metrics = {
        "n_samples": float(args.n_samples),
        "teacher_width_mae_px": teacher_width_mae,
        "teacher_position_mae_mm": teacher_pos_mae,
    }
    metrics.update(run_torch_demo(samples, teacher_w, args.epochs, args.seed))
    df = pd.DataFrame([metrics])
    df.to_csv(OUT / "synthetic_demo_metrics.csv", index=False)
    print(df.to_string(index=False))
    print(f"Synthetic demo outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
