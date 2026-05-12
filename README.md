# 高级机器学习理论课程报告配套程序（100分冲刺版）

题目：**基于确定性教师伪标签与轻量级 1D-CNN 的双通道锥光全息位移测量方法**

仓库链接：https://github.com/Michael326580/advanced-ml-theory-report-2026

## 1. 项目简介

本仓库用于华中科技大学 2026 年研究生《高级机器学习理论》课程报告提交。项目面向工业精密制造中的锥光全息双通道线性位移测量问题：传感器输出为 `2×2048` 双通道一维干涉条纹波形，目标是估计条纹宽度 `W (pixel)`，再通过有理标定模型重构线性位移 `x (mm)`。

课程报告中的算法思路为：

```text
双通道波形采集
→ 确定性教师 PCCC + FFT 生成 W(pixel) 伪标签
→ 轻量级 1D-CNN plain_cnn_w24 学习 waveform → W_hat
→ 有理标定模型 x_hat = a/(W_hat-b)+c 重构位移
→ 精度、时延、复杂度、鲁棒性和补充验证分析
```

选题方向：**提出了创新性的算法思路解决实际问题**。

## 2. 课程评分对应关系

本仓库按“满分准备、可复现提交”的标准整理：

| 课程评分点 | 仓库对应内容 |
|---|---|
| 方法新颖性 | 确定性教师伪标签 + 轻量 1D-CNN + 显式物理标定 |
| 方法介绍 | 报告正文 + `synthetic_waveform_demo.py` 流程演示 |
| 实验设计 | 主结果、消融实验、鲁棒性实验、补充验证实验 |
| 实验分析 | `outputs/reproduced_summary.md` 与 `outputs/extended_validation_summary.md` |
| 程序可运行 | `run_all.py` 与 `tests/smoke_test.py` |
| 环境说明 | 本 README 与 `requirements.txt` |
| 结果一致性 | `data/*.csv` → `outputs/*_reproduced.csv` |

> 说明：课程要求中总分超过 99 分会截断为 99 分，因此这里的“100分冲刺版”指按满分材料准备，而不是承诺实际评分。

## 3. 仓库结构

```text
advanced-ml-theory-report-2026/
├── README.md
├── requirements.txt
├── requirements-torch.txt
├── reproduce_report_results.py
├── synthetic_waveform_demo.py
├── extended_validation_experiments.py
├── run_all.py
├── data/
│   ├── dataset_summary.csv
│   ├── main_comparison.csv
│   ├── fast_student_ablation.csv
│   ├── robustness_results.csv
│   └── calibration.json
├── outputs/
│   └── 运行脚本后生成复现表格、图表和验证摘要
├── tests/
│   └── smoke_test.py
└── report/
    └── 高级机器学习理论课程报告_陈卓_100分冲刺版.md
```

## 4. 环境配置

推荐 Python 3.10 或更高版本。

基础依赖安装：

```bash
pip install -r requirements.txt
```

基础依赖包括：

```text
numpy
pandas
matplotlib
scipy
python-docx
```

如果希望运行 `synthetic_waveform_demo.py` 中的可选 CNN 训练演示，可额外安装 PyTorch：

```bash
pip install -r requirements-torch.txt
```

如果未安装 PyTorch，`synthetic_waveform_demo.py` 会自动跳过 CNN 训练，仅运行 FFT teacher 演示，不影响主结果复现。

## 5. 一键运行

在仓库根目录执行：

```bash
python run_all.py
```

该命令会依次运行：

```bash
python reproduce_report_results.py
python synthetic_waveform_demo.py --n-samples 120 --epochs 2
python extended_validation_experiments.py
```

## 6. 分步运行与验证

### 6.1 复现报告核心表格和图

```bash
python reproduce_report_results.py
```

输出：

```text
outputs/dataset_summary_reproduced.csv
outputs/main_comparison_reproduced.csv
outputs/fast_student_ablation_reproduced.csv
outputs/robustness_results_reproduced.csv
outputs/accuracy_latency_tradeoff.png
outputs/fast_student_ablation.png
outputs/robustness_summary.png
outputs/reproduced_metrics.json
outputs/reproduced_summary.md
```

该脚本对应课程报告第 5 章“实验结果”。

### 6.2 运行合成波形算法演示

```bash
python synthetic_waveform_demo.py --n-samples 120 --epochs 5
```

输出：

```text
outputs/synthetic_waveforms.png
outputs/synthetic_waveform_demo.png
outputs/synthetic_demo_metrics.csv
outputs/synthetic_demo_metrics.json
```

该脚本用于演示“合成双通道反相信号 → 简化 FFT teacher → W 估计 → 可选 CNN 回归”的方法流程。其结果不作为真实工业实验结论。

### 6.3 运行补充验证实验

```bash
python extended_validation_experiments.py
```

输出：

```text
outputs/tradeoff_audit.csv
outputs/tradeoff_audit.png
outputs/robustness_improvement_audit.csv
outputs/robustness_improvement_audit.png
outputs/calibration_sensitivity.csv
outputs/calibration_sensitivity.png
outputs/synthetic_noise_sweep.csv
outputs/synthetic_noise_sweep.png
outputs/extended_validation_summary.json
outputs/extended_validation_summary.md
```

补充验证包括：

1. **精度-时延折中复核**：验证 `plain_cnn_w24` 在 w16/w24/w32 中为综合折中最优；
2. **鲁棒性改善比例计算**：量化 robust_w24 在 Gaussian noise 和 channel shift 下的改善，同时识别 failure cases；
3. **标定灵敏度分析**：分析 `|dx/dW|`，说明小的像素宽度误差为何会放大成位移误差；
4. **合成噪声扫描**：在合成双通道条纹上验证 FFT teacher 随噪声增强的退化趋势。

## 7. Smoke test：证明代码可运行

课程提交前建议执行：

```bash
python tests/smoke_test.py
```

若成功，会看到：

```text
Smoke test 通过：课程报告配套程序可以正常运行。
```

建议保留以下截图作为提交备份：

1. `pip install -r requirements.txt` 安装成功；
2. `python run_all.py` 或 `python tests/smoke_test.py` 运行成功；
3. `outputs/` 文件夹生成图表和 CSV。

## 8. 报告核心结果

### 8.1 主结果

| Method | MAE (mm) | RMSE (mm) | MaxAE (mm) | Mean latency |
|---|---:|---:|---:|---|
| Teacher/PCCC reference | 0.0678 | 0.0858 | 0.2999 | 1.1574 ms/segment core |
| Attention student | 0.1166 | 0.1530 | 0.5641 | 10.2508 ms/sample |
| Plain FFT table | 0.1761 | 0.2088 | 0.7149 | 3.1660 ms/sample |
| Clean plain_cnn_w24 | 0.0994 | 0.1410 | 0.6236 | 0.8898 ms/sample |
| Robust plain_cnn_w24 ablation | 0.1046 | 0.1447 | 0.6341 | 0.9141 ms/sample |

### 8.2 Compact student 消融

| Model | Params | MACs (M) | MAE (mm) | RMSE (mm) | Mean latency |
|---|---:|---:|---:|---:|---:|
| plain_cnn_w16 | 26,817 | 6.324 | 0.1307 | 0.1842 | 0.6566 ms |
| plain_cnn_w24 | 59,809 | 14.008 | 0.0994 | 0.1410 | 0.8898 ms |
| plain_cnn_w32 | 105,857 | 24.707 | 0.1046 | 0.1451 | 1.1756 ms |

结论：w16 最快但误差较大；w24 综合最优；w32 参数量和 MACs 增大，但 MAE 未优于 w24。

### 8.3 鲁棒性结论

robust_w24 在 Gaussian noise 和 small channel shift 下明显改善 clean_w24 的 OOD 敏感性；但 amplitude scaling 和 spike noise 仍是失败模式。因此报告中不能写“robust_w24 在所有扰动下优于 FFT proxy”。

## 9. 结果边界

1. `Teacher/PCCC reference` 是确定性教师算法和标签生成参考，不是独立真实值；
2. `plain_cnn_w24` 的 0.8898 ms/sample 是本地 CPU、single-thread、FP32、batch=1 timing；
3. Teacher/PCCC 的 1.1574 ms/segment 是 core timing reference，不代表完整端到端系统；
4. 有理标定模型 R²=0.999967 不能等同独立 ground truth；
5. RMSE 是验证集预测误差，不是完整 GUM 标准不确定度；
6. `synthetic_waveform_demo.py` 和 `synthetic_noise_sweep` 是算法流程/补充验证实验，不替代真实工业采集实验；
7. 原始工业波形数据因项目保密原因不随课程包公开。

## 10. 与完整项目的关系

完整研究项目仓库：

```text
https://github.com/Michael326580/deep-study
```

完整项目包含教师标签生成、异常修复、蒸馏数据集构建、学生网络训练、鲁棒性实验和论文图表生成等脚本。本仓库是为了课程提交和快速复核整理的精简可运行版本。
