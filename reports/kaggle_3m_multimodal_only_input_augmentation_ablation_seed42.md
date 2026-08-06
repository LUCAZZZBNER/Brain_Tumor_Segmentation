# Kaggle 3M 纯多通道患者队列输入与增强 2×2 消融报告

生成时间：2026-08-05T01:13:43+08:00。

## 1. 实验目的

以 E0 单通道 FLAIR、无增强 U-Net 为 baseline，在相同 clean patient-level split 上分别验证三通道多模态输入（A）、轻量数据增强（B）及二者交互作用。

## 2. 数据与实验定义

使用 104 名患者、3629 张切片；测试集包含 10 名患者、525 张切片。六名单通道患者已按患者级排除。

| 模型 | 输入 | 增强 | 配置 |
|---|---|---|---|
| E0 | 单通道 | 无 | [configs/kaggle_3m_multimodal_only_e0_flair_unet.yaml](../configs/kaggle_3m_multimodal_only_e0_flair_unet.yaml) |
| E1-A | 三通道 | 无 | [configs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation.yaml](../configs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation.yaml) |
| E2-B | 单通道 | 有 | [configs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation.yaml](../configs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation.yaml) |
| M0-AB | 三通道 | 有 | [configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml](../configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml) |

## 3. 训练与验证结果

| 模型 | 最佳 Epoch | Train Positive IoU | Val Positive IoU | Train–Val Gap | 时间 |
|---|---:|---:|---:|---:|---:|
| E0 | 32 | 0.8056 | 0.5220 | 0.2835 | 25.5 min |
| E1-A | 31 | 0.8024 | 0.6645 | 0.1379 | 24.9 min |
| E2-B | 48 | 0.8231 | 0.5676 | 0.2555 | 31.3 min |
| M0-AB | 19 | 0.6963 | 0.6762 | 0.0202 | 20.4 min |

## 4. 测试集结果

| 模型 | Positive IoU | Positive Dice | Micro IoU | Precision | Recall | 空切片误报率 | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.6382 | 0.7228 | 0.6559 | 0.8583 | 0.7356 | 21.31% | 0.0790 |
| E1-A | 0.6442 | 0.7178 | 0.6944 | 0.8888 | 0.7605 | 13.35% | 0.0785 |
| E2-B | 0.7135 | 0.7952 | 0.7652 | 0.9190 | 0.8206 | 7.67% | 0.0577 |
| M0-AB | 0.6336 | 0.7238 | 0.5218 | 0.6726 | 0.6995 | 22.44% | 0.0833 |

## 5. 独立效应与交互作用

| 对比 | 含义 | Positive IoU 变化 |
|---|---|---:|
| E1-A − E0 | 三通道输入独立效应 | +0.0060 |
| E2-B − E0 | 数据增强独立效应 | +0.0753 |
| M0-AB − E0 | 三通道与增强总变化 | -0.0046 |
| M0-AB − E1-A − E2-B + E0 | A×B 交互作用 | -0.0859 |

## 6. 不同病灶面积的 Positive IoU

| 模型 | 微小 1–255 px | 小型 256–1023 px | 中型 1024–4095 px | 大型 ≥4096 px |
|---|---:|---:|---:|---:|
| E0 | 0.2952 (n=12) | 0.5359 (n=38) | 0.6866 (n=103) | 0.7892 (n=20) |
| E1-A | 0.1756 (n=12) | 0.4986 (n=38) | 0.7200 (n=103) | 0.8119 (n=20) |
| E2-B | 0.3314 (n=12) | 0.5935 (n=38) | 0.7754 (n=103) | 0.8523 (n=20) |
| M0-AB | 0.2361 (n=12) | 0.5331 (n=38) | 0.7173 (n=103) | 0.6324 (n=20) |

## 7. 自动结论

三通道输入单独使用使 Test Positive Macro IoU 提升 0.0060。
轻量数据增强单独使用使 Test Positive Macro IoU 提升 0.0753。
三通道与增强组合使 Test Positive Macro IoU 下降 0.0046。
A×B 交互作用使 Test Positive Macro IoU 下降 0.0859。

本报告来自 seed 42 单次实验，正式结论需要在相同 clean split 上追加 seed 123 和 2026，并报告 mean ± std。

## 8. 原始结果

- E0：[训练总结](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed42/test_metrics.json)
- E1-A：[训练总结](../runs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation_seed42/test_metrics.json)
- E2-B：[训练总结](../runs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation_seed42/test_metrics.json)
- M0-AB：[训练总结](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/test_metrics.json)
