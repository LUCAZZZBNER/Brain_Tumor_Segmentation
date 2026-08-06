# Kaggle 3M 纯多通道患者队列六模型多随机种子消融报告

生成时间：2026-08-05T19:04:37+08:00。

## 1. 实验范围

本报告汇总训练随机种子 42, 123, 2026。所有运行使用同一个 clean patient-level manifest，只改变训练随机种子和输出目录。

数据包含 104 名患者、3629 张切片；测试集包含 10 名患者、525 张切片。

## 2. 实验矩阵

| 模型 | 定义 |
|---|---|
| E0 | 单通道 FLAIR，无增强，普通 U-Net |
| E1-A | 三通道多模态，无增强，普通 U-Net |
| E2-B | 单通道 FLAIR，轻量增强，普通 U-Net |
| M0-AB | 三通道多模态，轻量增强，普通 U-Net |
| M4-NP | 三通道多模态，轻量增强，ResNet34，无预训练 |
| M4-P | 三通道多模态，轻量增强，ResNet34，ImageNet 预训练 |

## 3. 各随机种子 Test Positive Macro IoU

| 模型 | Seed 42 | Seed 123 | Seed 2026 | Mean ± SD |
|---|---:|---:|---:|---:|
| E0 | 0.6382 | 0.6515 | 0.6708 | 0.6535 ± 0.0164 |
| E1-A | 0.6442 | 0.6960 | 0.6484 | 0.6629 ± 0.0287 |
| E2-B | 0.7135 | 0.7160 | 0.7024 | 0.7106 ± 0.0072 |
| M0-AB | 0.6336 | 0.7097 | 0.7448 | 0.6961 ± 0.0568 |
| M4-NP | 0.7606 | 0.7603 | 0.7583 | 0.7597 ± 0.0013 |
| M4-P | 0.7753 | 0.7656 | 0.7582 | 0.7664 ± 0.0086 |

## 4. 测试指标 Mean ± SD

| 模型 | Positive IoU | Positive Dice | Micro IoU | Precision | Recall | 空切片误报率 |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0.6535 ± 0.0164 | 0.7342 ± 0.0147 | 0.6820 ± 0.0228 | 0.8892 ± 0.0295 | 0.7453 ± 0.0152 | 15.8144 ± 6.2003% |
| E1-A | 0.6629 ± 0.0287 | 0.7410 ± 0.0343 | 0.6914 ± 0.0044 | 0.8683 ± 0.0338 | 0.7739 ± 0.0307 | 14.2045 ± 5.8704% |
| E2-B | 0.7106 ± 0.0072 | 0.7962 ± 0.0074 | 0.7326 ± 0.0491 | 0.8793 ± 0.0830 | 0.8167 ± 0.0167 | 14.4886 ± 14.5996% |
| M0-AB | 0.6961 ± 0.0568 | 0.7804 ± 0.0504 | 0.6466 ± 0.1214 | 0.7673 ± 0.1109 | 0.7976 ± 0.0850 | 22.1591 ± 12.6444% |
| M4-NP | 0.7597 ± 0.0013 | 0.8356 ± 0.0053 | 0.7969 ± 0.0254 | 0.8869 ± 0.0201 | 0.8868 ± 0.0149 | 12.0265 ± 1.4578% |
| M4-P | 0.7664 ± 0.0086 | 0.8425 ± 0.0101 | 0.7926 ± 0.0153 | 0.8710 ± 0.0058 | 0.8983 ± 0.0239 | 18.1818 ± 2.2188% |

## 5. 训练与验证稳定性

| 模型 | Best Epoch | Val Positive IoU | Train–Val Gap | 训练时间 |
|---|---:|---:|---:|---:|
| E0 | 33.3333 ± 1.5275 | 0.5247 ± 0.0115 | 0.3120 ± 0.0269 | 25.8890 ± 0.6805 min |
| E1-A | 36.3333 ± 4.7258 | 0.6470 ± 0.0405 | 0.1918 ± 0.0621 | 26.8739 ± 1.7298 min |
| E2-B | 61.0000 ± 44.9333 | 0.5807 ± 0.0114 | 0.2197 ± 0.0865 | 36.5209 ± 17.9760 min |
| M0-AB | 34.3333 ± 15.5027 | 0.7050 ± 0.0285 | 0.0544 ± 0.0360 | 26.3714 ± 6.2483 min |
| M4-NP | 60.6667 ± 10.7858 | 0.7278 ± 0.0272 | 0.0953 ± 0.0247 | 44.5344 ± 5.1722 min |
| M4-P | 28.0000 ± 5.1962 | 0.7547 ± 0.0058 | 0.0573 ± 0.0115 | 28.6224 ± 2.3717 min |

## 6. 逐随机种子配对消融效应

| 对比 | Seed 42 | Seed 123 | Seed 2026 | Mean ± SD |
|---|---:|---:|---:|---:|
| A：E1-A − E0 | +0.0060 | +0.0445 | -0.0225 | 0.0094 ± 0.0336 |
| B：E2-B − E0 | +0.0753 | +0.0645 | +0.0316 | 0.0571 ± 0.0228 |
| A+B：M0-AB − E0 | -0.0046 | +0.0583 | +0.0740 | 0.0426 ± 0.0416 |
| A×B 交互 | -0.0859 | -0.0508 | +0.0649 | -0.0240 ± 0.0789 |
| 结构：M4-NP − M0-AB | +0.1270 | +0.0506 | +0.0134 | 0.0637 ± 0.0579 |
| 预训练：M4-P − M4-NP | +0.0147 | +0.0053 | -0.0000 | 0.0066 ± 0.0075 |
| 总变化：M4-P − E0 | +0.1371 | +0.1141 | +0.0874 | 0.1129 ± 0.0249 |

## 7. 综合排名与结论

1. M4-P：平均 Test Positive Macro IoU 0.7664
2. M4-NP：平均 Test Positive Macro IoU 0.7597
3. E2-B：平均 Test Positive Macro IoU 0.7106
4. M0-AB：平均 Test Positive Macro IoU 0.6961
5. E1-A：平均 Test Positive Macro IoU 0.6629
6. E0：平均 Test Positive Macro IoU 0.6535

只有当逐 seed 配对效应方向一致、均值大于其随机波动时，才应将对应因素写成稳定结论。不得从三个 seed 中挑选最高一次作为最终结果。

多随机种子只评估固定患者划分上的训练随机性；测试集仍只有少量患者，最终模型还需要 patient-level 交叉验证或外部验证。

## 8. 原始结果

### E0

- Seed 42：[训练总结](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed42/test_metrics.json)
- Seed 123：[训练总结](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed123/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed123/test_metrics.json)
- Seed 2026：[训练总结](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed2026/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed2026/test_metrics.json)

### E1-A

- Seed 42：[训练总结](../runs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation_seed42/test_metrics.json)
- Seed 123：[训练总结](../runs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation_seed123/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation_seed123/test_metrics.json)
- Seed 2026：[训练总结](../runs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation_seed2026/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation_seed2026/test_metrics.json)

### E2-B

- Seed 42：[训练总结](../runs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation_seed42/test_metrics.json)
- Seed 123：[训练总结](../runs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation_seed123/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation_seed123/test_metrics.json)
- Seed 2026：[训练总结](../runs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation_seed2026/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation_seed2026/test_metrics.json)

### M0-AB

- Seed 42：[训练总结](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/test_metrics.json)
- Seed 123：[训练总结](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed123/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed123/test_metrics.json)
- Seed 2026：[训练总结](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed2026/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed2026/test_metrics.json)

### M4-NP

- Seed 42：[训练总结](../runs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed42/test_metrics.json)
- Seed 123：[训练总结](../runs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed123/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed123/test_metrics.json)
- Seed 2026：[训练总结](../runs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed2026/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed2026/test_metrics.json)

### M4-P

- Seed 42：[训练总结](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42/test_metrics.json)
- Seed 123：[训练总结](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed123/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed123/test_metrics.json)
- Seed 2026：[训练总结](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed2026/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed2026/test_metrics.json)
