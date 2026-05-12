# 高级机器学习理论课程报告配套程序

题目：基于确定性教师伪标签与轻量级 1D-CNN 的双通道锥光全息位移测量方法

本程序包用于课程报告提交，包含两部分：

1. `reproduce_report_results.py`：读取随包提供的审计结果 CSV，复现报告中的核心表格与图，包括主结果对比、fast student 消融和鲁棒性分析。
2. `synthetic_waveform_demo.py`：在不公开原始工业采集数据的前提下，生成一组双通道干涉波形样例，演示“教师 FFT 宽度估计 + 轻量 1D-CNN 回归”的基本流程。若本机未安装 PyTorch，则自动跳过 CNN 训练，仅执行 FFT 教师算法演示。

## 环境配置

推荐 Python 3.10+。安装依赖：

```bash
pip install -r requirements.txt
```

如果只运行结果复现脚本，必须依赖为 `numpy/pandas/matplotlib`；`torch` 仅用于 CNN 演示。

## 运行方式

在本目录下执行：

```bash
python reproduce_report_results.py
python synthetic_waveform_demo.py --n-samples 120 --epochs 5
```

## 输出文件

运行后会生成：

```text
outputs/
├── main_comparison_reproduced.csv
├── fast_student_ablation_reproduced.csv
├── robustness_results_reproduced.csv
├── accuracy_latency_tradeoff.png
├── robustness_summary.png
├── synthetic_waveforms.png
└── synthetic_demo_metrics.csv
```

## 与完整 GitHub 项目的关系

完整项目仓库为：

```text
https://github.com/Michael326580/deep-study
```

完整仓库包含教师标签生成、异常修复、蒸馏数据集构建、学生网络训练、评估诊断和论文图表生成等脚本。本课程包是为了课程提交和快速复核而整理的精简可运行版本；原始工业采集数据不随课程包公开。

## 结果边界

报告中的主结果来自完整项目的最终审计材料。本精简包中的 `data/*.csv` 保存这些审计指标，可复现报告表格和图。`synthetic_waveform_demo.py` 只用于演示算法流程，不能替代真实实验结果。
