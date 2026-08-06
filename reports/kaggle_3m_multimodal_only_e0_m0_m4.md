# Kaggle 3M 纯多通道队列 E0、M0、M4 对比报告

生成时间：2026-08-04T21:17:57+08:00。

## 1. 实验目的

本实验从 Kaggle 3M 队列中按患者级排除包含灰度等价切片（R=G=B）的患者，保留其余患者原有的 train/val/test 归属，然后从头训练并测试 E0、M0 和 M4。除数据清理、输出目录和实验名称外，各模型设置与原实验保持一致。

## 2. 数据清理结果

- 保留患者：104 名
- 保留切片：3629 张
- 排除患者：6 名
- 排除切片：300 张
- 划分策略：过滤原 manifest，未重新随机分配患者

| Split | 患者 | 切片 | 正切片 | 空切片 |
|---|---:|---:|---:|---:|
| train | 85 | 2619 | 935 | 1684 |
| val | 9 | 485 | 167 | 318 |
| test | 10 | 525 | 173 | 352 |

排除患者明细：

| 患者 | 原 Split | 切片 | 正切片 | 空切片 |
|---|---|---:|---:|---:|
| TCGA_DU_7013_19860523 | train | 49 | 12 | 37 |
| TCGA_FG_A60K_20040224 | val | 73 | 24 | 49 |
| TCGA_HT_7877_19980917 | train | 30 | 9 | 21 |
| TCGA_HT_8105_19980826 | train | 32 | 17 | 15 |
| TCGA_HT_A616_19991226 | val | 28 | 11 | 17 |
| TCGA_HT_A61B_19991127 | test | 88 | 25 | 63 |

## 3. 实验定义

| 编号 | 模型 | 配置 |
|---|---|---|
| E0 | 单通道 FLAIR U-Net（无增强） | [configs/kaggle_3m_multimodal_only_e0_flair_unet.yaml](../configs/kaggle_3m_multimodal_only_e0_flair_unet.yaml) |
| M0 | 三通道多模态 U-Net（轻量增强） | [configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml](../configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml) |
| M4 | 三通道多模态 ResNet34 U-Net（轻量增强） | [configs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet.yaml](../configs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet.yaml) |

三组实验均使用随机种子 42、patient-level 固定划分、256×256 输入、batch size 4、AdamW、BCE + Positive Dice loss，并以 Validation Positive Macro IoU 选择最佳 checkpoint。M0 与 M4 使用相同的水平翻转和 ±5° 旋转。

## 4. 训练与验证结果

| 模型 | 最佳 Epoch | Train Positive IoU | Val Positive IoU | Train–Val Gap | 训练时间 |
|---|---:|---:|---:|---:|---:|
| E0 | 32 | 0.8056 | 0.5220 | 0.2835 | 25.5 min |
| M0 | 19 | 0.6963 | 0.6762 | 0.0202 | 20.4 min |
| M4 | 25 | 0.7999 | 0.7558 | 0.0441 | 27.3 min |

## 5. 测试集结果

| 模型 | Positive Macro IoU | Positive Macro Dice | Micro IoU | Precision | Recall | 空切片误报率 | Test Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.6382 | 0.7228 | 0.6559 | 0.8583 | 0.7356 | 21.31% | 0.0790 |
| M0 | 0.6336 | 0.7238 | 0.5218 | 0.6726 | 0.6995 | 22.44% | 0.0833 |
| M4 | 0.7753 | 0.8528 | 0.8015 | 0.8735 | 0.9068 | 15.62% | 0.0429 |

按 Test Positive Macro IoU 排名：

1. M4：0.7753
2. E0：0.6382
3. M0：0.6336

## 6. 相对原混合队列的变化

下表用于观察清除单通道患者后重新训练的变化。由于测试集由 11 名患者变为 10 名患者，同时模型也从头训练，因此该变化不能只解释为删除某一患者的直接贡献。

| 模型 | 原 Positive IoU | 清理后 Positive IoU | 变化 | 原空切片误报率 | 清理后误报率 | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0.6762 | 0.6382 | -0.0380 | 9.64% | 21.31% | +11.67 pp |
| M0 | 0.7192 | 0.6336 | -0.0856 | 10.60% | 22.44% | +11.84 pp |
| M4 | 0.7471 | 0.7753 | +0.0282 | 16.87% | 15.62% | -1.24 pp |

## 7. 自动结论

在纯多通道患者队列上，M4 获得最高 Test Positive Macro IoU：0.7753。

本报告的模型选择以 Positive Macro IoU 为主，同时应结合 Recall、Precision 和空切片误报率判断临床取舍。所有结果仍来自单次 seed 42 实验，后续显著性结论需要额外随机种子或 patient-level 交叉验证。

## 8. 原始结果

- E0：[训练总结](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed42/test_metrics.json)
- M0：[训练总结](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/test_metrics.json)
- M4：[训练总结](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42/test_metrics.json)
- [纯多通道 split 元数据](../splits/kaggle_3m_multimodal_only_seed42.meta.json)
