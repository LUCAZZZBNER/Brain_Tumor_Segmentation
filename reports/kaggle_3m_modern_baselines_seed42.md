# Kaggle 3M 清洗患者集：Seed-42 现代基线报告

生成时间（UTC）：2026-08-19T03:03:34+00:00

## 实验一致性

- 患者级固定划分：训练 85、验证 9、测试 10，同一患者不会跨集合。
- 切片数：训练 2619、验证 485、测试 525。
- 随机种子：42；清洗后 manifest SHA-256：`6aee075d9ea8b4a41cdefc372e381a55ec3a099acfaf16002dbde2cd7d5051e7`。
- nnU-Net 和 TransUNet 均禁用训练时数据增强；测试集仅用于最终冻结评估。

## 测试集结果

| 模型 | 状态 | Positive Macro IoU | Positive Macro Dice | Micro IoU | Micro Precision | Micro Recall | Empty-slice FPR | ΔIoU vs M0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean M0 U-Net (reference) | 完成 | 0.6336 | 0.7238 | 0.5218 | 0.6726 | 0.6995 | 0.2244 | +0.0000 |
| M4-P / clean ResNet34 U-Net | 完成 | 0.7753 | 0.8528 | 0.8015 | 0.8735 | 0.9068 | 0.1562 | +0.1417 |
| Official nnU-Net v2 2D (NoDA) | 完成 | 0.6721 | 0.7597 | 0.7028 | 0.9362 | 0.7381 | 0.0312 | +0.0384 |
| Basic TransUNet 2D | 完成 | 0.6823 | 0.7672 | 0.6336 | 0.7248 | 0.8343 | 0.2330 | +0.0486 |

## 现代基线运行配置

| 项目 | nnU-Net v2 2D | Basic TransUNet 2D |
| --- | --- | --- |
| 输入 | 3 通道，256×256 | 3 通道，256×256 |
| 初始化 | nnU-Net 官方随机初始化 | 随机初始化 |
| 数据增强 | `nnUNetTrainerNoDA`，禁用 | 禁用 |
| Batch size | 4（仅资源覆盖；官方计划值保留） | 2 |
| 数据进程 | 0（官方单线程 augmenter） | 0（Windows 安全设置） |
| 推理预处理/导出进程 | 1 / 1 | 不适用 |
| 测试增强 | 禁用 TTA | 禁用 |

## 可重复性与解释说明

- 表中只有 manifest hash 与 525 张固定测试切片同时匹配的结果才会显示指标并参与 ΔIoU 计算。
- `Clean M0 U-Net` 是当前清洗患者集上的同清单参考。旧 E0 使用清洗前 manifest，不能与这里的现代基线直接计算差值。
- nnU-Net 的 batch size 4、数据进程 0 和导出进程 1 是 Windows 资源安全设置；网络、损失、样本、固定划分、种子及 NoDA 策略不变。
- `Basic TransUNet 2D` 是从零训练的基础对照，并非论文原始的 R50-ViT-B/16 预训练版本；投稿表格中应明确这一点。

## 结果文件

- Clean M0 U-Net (reference)：`runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/test_metrics.json`（完成）
- M4-P / clean ResNet34 U-Net：`runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42/test_metrics.json`（完成）
- Official nnU-Net v2 2D (NoDA)：`runs/nnunetv2_2d_kaggle_3m_clean_no_augmentation_seed42/test_metrics.json`（完成）
- Basic TransUNet 2D：`runs/kaggle_3m_multimodal_only_transunet_2d_basic_no_augmentation_seed42/test_metrics.json`（完成）
