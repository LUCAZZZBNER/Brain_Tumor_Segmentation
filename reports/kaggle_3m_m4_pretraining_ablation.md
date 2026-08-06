# Kaggle 3M M4 ResNet34 预训练单变量消融报告

生成时间：2026-08-04T23:28:45+08:00。

## 1. 实验目的

本实验在纯多通道患者队列上分离 ResNet34 encoder 结构与 ImageNet 预训练的贡献。M4-NP 与 M4-P 的唯一差异是 pretrained=false/true；其余数据、增强、损失函数、优化器、随机种子和训练策略完全一致。

## 2. 数据与实验定义

共使用 104 名患者、3629 张切片；测试集包含 10 名患者、525 张切片。

| 模型 | Encoder | ImageNet 预训练 | 配置 |
|---|---|---|---|
| M0 | DoubleConv | 否 | [configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml](../configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml) |
| M4-NP | ResNet34 | 否 | [configs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet.yaml](../configs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet.yaml) |
| M4-P | ResNet34 | 是 | [configs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet.yaml](../configs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet.yaml) |

## 3. 训练与验证结果

| 模型 | 最佳 Epoch | Train Positive IoU | Val Positive IoU | Train–Val Gap | 时间 |
|---|---:|---:|---:|---:|---:|
| M0 | 19 | 0.6963 | 0.6762 | 0.0202 | 20.4 min |
| M4-NP | 56 | 0.8196 | 0.7338 | 0.0858 | 42.7 min |
| M4-P | 25 | 0.7999 | 0.7558 | 0.0441 | 27.3 min |

## 4. 测试集结果

| 模型 | Positive IoU | Positive Dice | Micro IoU | Precision | Recall | 空切片误报率 | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.6336 | 0.7238 | 0.5218 | 0.6726 | 0.6995 | 22.44% | 0.0833 |
| M4-NP | 0.7606 | 0.8319 | 0.8216 | 0.9001 | 0.9040 | 10.80% | 0.0482 |
| M4-P | 0.7753 | 0.8528 | 0.8015 | 0.8735 | 0.9068 | 15.62% | 0.0429 |

## 5. 单变量贡献

| 对比 | 含义 | Positive IoU 变化 |
|---|---|---:|
| M4-NP − M0 | ResNet34 结构贡献 | +0.1270 |
| M4-P − M4-NP | ImageNet 预训练贡献 | +0.0147 |
| M4-P − M0 | 结构与预训练的总变化 | +0.1417 |

这些差值是模型级消融结果，不应解释为严格可加的因果效应。

## 6. 患者级 Positive IoU

| 患者 | M0 | M4-NP | M4-P | M4-NP − M0 | M4-P − M4-NP |
|---|---:|---:|---:|---:|---:|
| TCGA_DU_5872_19950223 | 0.5071 | 0.8868 | 0.9024 | +0.3797 | +0.0156 |
| TCGA_DU_6399_19830416 | 0.8863 | 0.8907 | 0.8922 | +0.0043 | +0.0015 |
| TCGA_DU_6405_19851005 | 0.0580 | 0.4866 | 0.5282 | +0.4287 | +0.0416 |
| TCGA_DU_6407_19860514 | 0.7884 | 0.7902 | 0.8391 | +0.0018 | +0.0489 |
| TCGA_DU_6408_19860521 | 0.6849 | 0.7859 | 0.7916 | +0.1011 | +0.0056 |
| TCGA_FG_8189_20030516 | 0.6587 | 0.7561 | 0.7795 | +0.0974 | +0.0234 |
| TCGA_FG_A4MU_20030903 | 0.6516 | 0.8453 | 0.8417 | +0.1937 | -0.0036 |
| TCGA_HT_7616_19940813 | 0.5285 | 0.6287 | 0.6444 | +0.1002 | +0.0157 |
| TCGA_HT_7693_19950520 | 0.8049 | 0.8205 | 0.7915 | +0.0156 | -0.0290 |
| TCGA_HT_7881_19981015 | 0.5358 | 0.5374 | 0.5639 | +0.0016 | +0.0265 |

## 7. 自动结论

ResNet34 结构与 ImageNet 预训练均带来正向提升。

本实验仍为 seed 42 单次运行。确定最终论文结论前，应在固定 clean split 上追加多个训练随机种子，并报告 mean ± std。

## 8. 原始结果

- M0：[训练总结](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/test_metrics.json)
- M4-NP：[训练总结](../runs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed42/test_metrics.json)
- M4-P：[训练总结](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42/training_summary.json)；[测试指标](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42/test_metrics.json)
