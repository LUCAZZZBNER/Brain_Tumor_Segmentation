# 脑肿瘤 MRI 二维分割项目说明

## 项目概述

本项目是一个面向科研实验的、可复现的二维脑肿瘤 MRI 分割系统。项目以像素级二值分割为目标：输入脑部 MRI 图像，输出肿瘤区域掩膜。整个流程覆盖数据审计、数据清洗、固定划分、模型训练、验证集模型选择、冻结测试集评估、单图推理和结果记录。

项目重点不是单纯追求最高分，而是保证实验过程可追溯、数据划分可复现、测试集不参与模型选择，并明确记录数据和方法的限制。

## 主要能力

- 对图像和掩膜进行配对、格式、尺寸、掩膜取值和重复样本检查。
- 使用带文件哈希和元数据哈希的固定数据清单，避免实验期间样本集合被悄悄改变。
- 支持按类别分层的 train/validation/test 划分，并检查重复样本是否跨集合泄漏。
- 提供 2D U-Net 基线以及 U-Net++、Attention U-Net、ASPP U-Net、ResNet34 U-Net 等模型变体。
- 支持 BCE、Soft Dice、Focal 和 Boundary Dice 等损失组合，以及轻量图像增强。
- 使用验证集的 macro IoU 选择 `best.pt`，测试集只用于最终冻结评估。
- 输出 IoU、Dice、Precision、Recall、分类别指标、逐样本结果和定性预测图。
- 支持从单张未标注图像生成二值掩膜和概率图。
- 提供 nnU-Net v2 与基础 TransUNet 的现代基线实验脚本。

## 数据集

`DATASET/` 保存本地数据，不纳入版本控制。项目当前包含以下数据及派生版本：

- `Segmentation/`：原始二维 MRI 分割数据。
- `kaggle_3m/`：Kaggle 3M 数据版本。
- `brisc2025/`：BRISC 2025 原始数据。
- `Segmentation_v2/`：经过重复样本和标签冲突清理的派生数据集。
- `BRISC2025_clean/`：从 BRISC 2025 数据中按保守规则清理得到的版本。

派生数据集的清单、隔离样本和审计摘要位于 `DATASET/` 根目录；固定划分文件位于 `splits/`。由于原始 PNG 数据没有可靠的患者 ID，现有协议只能保证样本级/相似性分组隔离，不能声称患者级独立性。

## 实验设计

核心实验集中在 `kaggle_3m_multimodal_only_*` 配置和运行目录中。主要比较因素包括：

1. 单通道 FLAIR 与三通道输入；
2. 无增强与轻量增强；
3. 标准 U-Net 与 ResNet34 编码器；
4. ImageNet 预训练与随机初始化；
5. 不同随机种子的重复实验；
6. nnU-Net v2 和基础 TransUNet 现代基线。

实验配置在 `configs/` 中，训练后的运行记录在本地 `runs/` 中。当前整理后的运行目录优先保留名称含 `multimodal_only` 的实验。

## 代码结构

```text
src/brain_tumor_seg/
├── config.py       配置读取与约束检查
├── data.py         数据集、掩膜处理和增强
├── splits.py       审计、分组和固定划分
├── make_splits.py  生成数据清单与划分
├── model.py        分割模型定义
├── losses.py       损失函数
├── metrics.py      IoU、Dice 及分类别指标
├── engine.py       训练和无梯度评估循环
├── train.py        训练入口
├── evaluate.py     验证集/测试集评估入口
├── predict.py      单图推理入口
├── reporting.py    训练曲线和结果报告
└── utils.py        随机种子、设备、日志和 checkpoint 工具
```

- `configs/`：不同数据、模型和消融实验的 YAML 配置。
- `scripts/`：数据构建、审计、批量训练、nnU-Net 及报告生成脚本。
- `tests/`：数据、划分、指标、模型和损失函数测试。
- `splits/`：冻结的数据划分清单及其元数据。
- `reports/`：审计结果、图表和最终文档材料。

## 典型运行流程

```powershell
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m pytest
python -m brain_tumor_seg.train --config configs/baseline.yaml --device auto
python -m brain_tumor_seg.evaluate --config configs/baseline.yaml --split test --device auto
python -m brain_tumor_seg.predict DATASET/Segmentation/Glioma/enh_1841.png --config configs/baseline.yaml --output-dir predictions
```

正式实验应先确认配置指向正确的数据清单和固定划分。模型选择、阈值搜索和早停只能使用训练集与验证集；测试集应在方案冻结后只评估一次。

## 可复现性与限制

- 配置中记录随机种子、数据路径、模型、损失、优化器、增强和评估策略。
- 清单保存图像与掩膜的 SHA-256，运行目录保存解析后的配置、环境和划分快照。
- `.gitignore` 排除本地数据、训练输出、预测结果和 Python 缓存，避免把大文件提交到仓库。
- 数据缺少可靠患者标识，因此不能把当前结果解释为患者级独立测试或外部泛化性能。
- 数据集类别和样本来源存在不完整或偏倚风险，论文和报告应同时说明这些限制。

## 环境要求

项目要求 Python 3.10 或更高版本，核心依赖包括 PyTorch、torchvision、NumPy、Pillow、PyYAML、matplotlib 和 tqdm。GPU 训练可使用 CUDA；没有 GPU 时可以回退到 CPU，但训练速度会明显降低。
