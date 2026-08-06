# Kaggle 3M 纯多通道患者队列完整消融报告（Seed 42）

## 1. 实验目的

本报告统一汇总 Kaggle 3M 纯多通道患者队列上的六组 seed 42 实验，以单通道 FLAIR、无数据增强的普通 U-Net（E0）为正式 baseline，依次分析：

- A：三通道多模态输入；
- B：轻量数据增强；
- A×B：三通道输入与数据增强的交互作用；
- C：ResNet34 encoder 结构；
- D：ImageNet 预训练。

所有实验使用相同的 patient-level 固定划分、256×256 输入、batch size 4、AdamW、`3e-4` 学习率、BCE + Positive Dice loss、最多 150 epochs、早停耐心 30，并以 Validation Positive Macro IoU 选择最佳 checkpoint。二值化阈值固定为 0.5。

## 2. 数据清理与划分

原 Kaggle 3M 队列包含 110 名患者、3929 张切片。审计发现 6 名患者的输入实际为灰度等价图像（R=G=B），因此按患者级整体排除。未物理删除原始文件，仅通过 clean manifest 过滤；其余患者保持原来的 train/val/test 归属，没有重新随机划分。

- 保留患者：104 名；
- 保留切片：3629 张；
- 排除患者：6 名；
- 排除切片：300 张。

| Split | 患者 | 切片 | 正切片 | 空切片 |
|---|---:|---:|---:|---:|
| Train | 85 | 2619 | 935 | 1684 |
| Validation | 9 | 485 | 167 | 318 |
| Test | 10 | 525 | 173 | 352 |

排除患者如下：

| 患者 | 原 Split | 切片 | 正切片 | 空切片 |
|---|---|---:|---:|---:|
| TCGA_DU_7013_19860523 | Train | 49 | 12 | 37 |
| TCGA_FG_A60K_20040224 | Validation | 73 | 24 | 49 |
| TCGA_HT_7877_19980917 | Train | 30 | 9 | 21 |
| TCGA_HT_8105_19980826 | Train | 32 | 17 | 15 |
| TCGA_HT_A616_19991226 | Validation | 28 | 11 | 17 |
| TCGA_HT_A61B_19991127 | Test | 88 | 25 | 63 |

本报告中的 E0/E2 虽然只读取绿色 FLAIR 通道，但仍使用同一 clean 患者队列，以保证六组实验面对完全相同的训练、验证和测试患者。

## 3. 完整实验矩阵

| 编号 | 输入 | 轻量增强 | Encoder | ImageNet 预训练 | 作用 |
|---|---|---|---|---|---|
| E0 | 单通道 FLAIR | 无 | DoubleConv U-Net | 否 | 正式 baseline |
| E1-A | 三通道多模态 | 无 | DoubleConv U-Net | 否 | 三通道输入独立效应 |
| E2-B | 单通道 FLAIR | 有 | DoubleConv U-Net | 否 | 数据增强独立效应 |
| M0-AB | 三通道多模态 | 有 | DoubleConv U-Net | 否 | A+B 总效果及交互作用 |
| M4-NP | 三通道多模态 | 有 | ResNet34 U-Net | 否 | ResNet34 结构贡献 |
| M4-P | 三通道多模态 | 有 | ResNet34 U-Net | 是 | ImageNet 预训练贡献 |

轻量增强统一为水平翻转（概率 0.5）和 ±5° 随机旋转。

严格比较关系为：

| 对比 | 解释 |
|---|---|
| E1-A − E0 | 三通道输入 A 的独立效应 |
| E2-B − E0 | 数据增强 B 的独立效应 |
| M0-AB − E1-A − E2-B + E0 | A×B 交互作用 |
| M4-NP − M0-AB | ResNet34 encoder 结构贡献 |
| M4-P − M4-NP | ImageNet 预训练贡献 |
| M4-P − E0 | 最终方案相对正式 baseline 的总变化 |

## 4. 训练与验证结果

| 模型 | 最佳 Epoch | Train Positive IoU | Val Positive IoU | Train–Val Gap | 训练时间 |
|---|---:|---:|---:|---:|---:|
| E0 | 32 | 0.8056 | 0.5220 | 0.2835 | 25.5 min |
| E1-A | 31 | 0.8024 | 0.6645 | 0.1379 | 24.9 min |
| E2-B | 48 | 0.8231 | 0.5676 | 0.2555 | 31.3 min |
| M0-AB | 19 | 0.6963 | 0.6762 | **0.0202** | 20.4 min |
| M4-NP | 56 | **0.8196** | 0.7338 | 0.0858 | 42.7 min |
| M4-P | 25 | 0.7999 | **0.7558** | 0.0441 | 27.3 min |

训练现象：

- M4-P 获得最高 Validation Positive IoU（0.7558），且 Train–Val gap 仅为 0.0441，泛化表现最好。
- M4-NP 的验证结果次高（0.7338），但训练时间最长，约为 42.7 分钟。
- E0 和 E2-B 的 Train–Val gap 分别为 0.2835 和 0.2555，单通道 U-Net 仍表现出明显过拟合。
- M0-AB 的 gap 最小不代表性能最好；其最佳点训练指标本身偏低，测试结果也明显落后。

## 5. 测试集总体结果

| 模型 | Positive Macro IoU | Positive Macro Dice | Micro IoU | Micro Precision | Micro Recall | 空切片误报率 | Test Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.6382 | 0.7228 | 0.6559 | 0.8583 | 0.7356 | 21.31% | 0.0790 |
| E1-A | 0.6442 | 0.7178 | 0.6944 | 0.8888 | 0.7605 | 13.35% | 0.0785 |
| E2-B | 0.7135 | 0.7952 | 0.7652 | **0.9190** | 0.8206 | **7.67%** | 0.0577 |
| M0-AB | 0.6336 | 0.7238 | 0.5218 | 0.6726 | 0.6995 | 22.44% | 0.0833 |
| M4-NP | 0.7606 | 0.8319 | **0.8216** | 0.9001 | 0.9040 | 10.80% | 0.0482 |
| M4-P | **0.7753** | **0.8528** | 0.8015 | 0.8735 | **0.9068** | 15.62% | **0.0429** |

按 Test Positive Macro IoU 排名：

1. M4-P：0.7753；
2. M4-NP：0.7606；
3. E2-B：0.7135；
4. E1-A：0.6442；
5. E0：0.6382；
6. M0-AB：0.6336。

不同指标对应不同取舍：

- M4-P 的 Positive Macro IoU、Positive Dice、Recall 和 Test Loss 最佳，是当前总体分割主模型。
- M4-NP 的 Micro IoU 更高，同时 Precision 更高、空切片误报率更低，是更保守且更均衡的 ResNet34 方案。
- E2-B 的 Precision 最高、空切片误报率最低，说明轻量增强对单通道 U-Net 非常有效，但其 Recall 和 Positive IoU 仍低于两个 ResNet34 模型。
- M0-AB 在 seed 42 上表现最差，说明三通道输入和增强的组合收益强烈依赖模型结构，不能假设它们在普通 U-Net 中一定协同。

## 6. 独立效应、交互作用与层级贡献

| 对比 | 含义 | Positive Macro IoU 变化 |
|---|---|---:|
| E1-A − E0 | 三通道输入独立效应 | +0.0060 |
| E2-B − E0 | 数据增强独立效应 | +0.0753 |
| M0-AB − E0 | 三通道与增强组合总变化 | -0.0046 |
| M0-AB − E1-A − E2-B + E0 | A×B 交互作用 | -0.0859 |
| M4-NP − M0-AB | ResNet34 结构贡献 | +0.1270 |
| M4-P − M4-NP | ImageNet 预训练贡献 | +0.0147 |
| M4-P − E0 | 最终方案相对 baseline 总变化 | **+0.1371** |

主要结论：

1. 三通道输入单独使用只有很小提升（+0.0060）。
2. 轻量数据增强是普通 U-Net 阶段最有效的独立因素（+0.0753）。
3. 在当前 seed 42 上，A×B 呈负交互（-0.0859），导致 M0-AB 略低于 E0；这一结论与清理前混合队列相反，必须通过多随机种子确认。
4. ResNet34 结构带来最大的层级提升：M4-NP 相对 M0-AB 提升 0.1270。
5. ImageNet 预训练进一步提升 Positive Macro IoU 0.0147，但同时 Precision 从 0.9001 降至 0.8735、空切片误报率从 10.80% 升至 15.62%。
6. 最终 M4-P 相对 E0 提升 0.1371，即 13.71 个百分点。

上述差值是模型级层级消融结果。除明确的单变量对比外，不应将其解释为严格可加的因果效应。

## 7. 不同病灶面积的 Positive IoU

| 模型 | 微小 1–255 px | 小型 256–1023 px | 中型 1024–4095 px | 大型 ≥4096 px |
|---|---:|---:|---:|---:|
| E0 | 0.2952 | 0.5359 | 0.6866 | 0.7892 |
| E1-A | 0.1756 | 0.4986 | 0.7200 | 0.8119 |
| E2-B | 0.3314 | 0.5935 | 0.7754 | 0.8523 |
| M0-AB | 0.2361 | 0.5331 | 0.7173 | 0.6324 |
| M4-NP | 0.2397 | 0.6375 | **0.8392** | **0.9022** |
| M4-P | **0.4359** | **0.6501** | 0.8383 | 0.8923 |

测试集病灶面积分布为：微小 12 张、小型 38 张、中型 103 张、大型 20 张。

- M4-P 在微小和小型病灶上最佳，尤其微小病灶 IoU 达到 0.4359。
- M4-NP 在中型和大型病灶上略优，分别达到 0.8392 和 0.9022。
- E2-B 在所有尺度上均明显优于 E0，进一步支持轻量增强对单通道 baseline 的有效性。
- E1-A 的微小病灶表现下降至 0.1756，说明仅增加通道可能削弱普通 U-Net 对微小病灶的稳定性。

## 8. 患者级 Positive IoU

| 测试患者 | E0 | E1-A | E2-B | M0-AB | M4-NP | M4-P |
|---|---:|---:|---:|---:|---:|---:|
| TCGA_DU_5872_19950223 | 0.3297 | 0.4392 | 0.5735 | 0.5071 | 0.8868 | **0.9024** |
| TCGA_DU_6399_19830416 | 0.8576 | **0.8925** | 0.8846 | 0.8863 | 0.8907 | 0.8922 |
| TCGA_DU_6405_19851005 | 0.2736 | 0.0105 | 0.3271 | 0.0580 | 0.4866 | **0.5282** |
| TCGA_DU_6407_19860514 | 0.7689 | 0.7369 | **0.8527** | 0.7884 | 0.7902 | 0.8391 |
| TCGA_DU_6408_19860521 | 0.7742 | 0.7910 | **0.7980** | 0.6849 | 0.7859 | 0.7916 |
| TCGA_FG_8189_20030516 | 0.6774 | 0.6480 | **0.7862** | 0.6587 | 0.7561 | 0.7795 |
| TCGA_FG_A4MU_20030903 | 0.6704 | 0.7691 | 0.7529 | 0.6516 | **0.8453** | 0.8417 |
| TCGA_HT_7616_19940813 | 0.5539 | 0.5347 | 0.6129 | 0.5285 | 0.6287 | **0.6444** |
| TCGA_HT_7693_19950520 | 0.7079 | 0.7830 | **0.8270** | 0.8049 | 0.8205 | 0.7915 |
| TCGA_HT_7881_19981015 | **0.5808** | 0.5216 | 0.5277 | 0.5358 | 0.5374 | 0.5639 |

患者级观察：

- M4-P 在全部 10 名测试患者上均高于 E0，说明总体提升并非只由单个患者驱动。
- M4-P 相对 M4-NP 在 8/10 名患者上更高，但两名患者出现轻微下降。
- E2-B 在 4 名患者上取得六模型最高值，说明单通道增强模型仍具有独立价值。
- 测试集只有 10 名患者，患者级表格用于描述差异，不能替代多随机种子或交叉验证。

## 9. Seed 42 总结

以 E0 为正式 baseline，seed 42 的完整层级消融支持以下结论：

> 在排除 6 名灰度等价患者、保持 patient-level 固定划分后，轻量数据增强是普通 U-Net 中最有效的独立因素；三通道输入单独只带来很小提升，并与增强在普通 U-Net 中呈负交互。将 encoder 替换为 ResNet34 后，三通道增强方案的分割性能显著提高；ImageNet 预训练进一步改善 Positive Macro IoU 和微小病灶表现。最终 M4-P 的 Test Positive Macro IoU 为 0.7753，较 E0 的 0.6382 提升 13.71 个百分点。

如果更重视最高总体分割 IoU 和微小病灶能力，当前推荐 M4-P；如果更重视 Precision、Micro IoU 和较低空切片误报，M4-NP 是更均衡的候选；如果计算资源有限且强调低误报，E2-B 是最强的普通 U-Net 方案。

## 10. 局限与下一步

- 六组结果均仅来自训练随机种子 42，尚不能判断 A×B 负交互和预训练 0.0147 的提升是否稳定。
- 测试集只有 10 名患者；多随机种子只能评估训练随机性，不能解决患者抽样不确定性。
- 下一步应保持同一 clean manifest，仅将训练种子改为 123 和 2026，对六组配置完整复现并报告 mean ± std。
- 多随机种子完成后，再对最终候选 E0、M4-NP、M4-P 进行 patient-level 交叉验证或外部验证。
- 不应从多个 seed 中挑选最高结果作为最终结果，应报告所有预设 seed 的平均值、标准差和逐 seed 配对差值。

## 11. 配置与原始结果

### E0

- [配置](../configs/kaggle_3m_multimodal_only_e0_flair_unet.yaml)
- [训练总结](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed42/training_summary.json)
- [测试指标](../runs/kaggle_3m_multimodal_only_e0_flair_unet_seed42/test_metrics.json)

### E1-A

- [配置](../configs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation.yaml)
- [训练总结](../runs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation_seed42/training_summary.json)
- [测试指标](../runs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation_seed42/test_metrics.json)

### E2-B

- [配置](../configs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation.yaml)
- [训练总结](../runs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation_seed42/training_summary.json)
- [测试指标](../runs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation_seed42/test_metrics.json)

### M0-AB

- [配置](../configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml)
- [训练总结](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/training_summary.json)
- [测试指标](../runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/test_metrics.json)

### M4-NP

- [配置](../configs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet.yaml)
- [训练总结](../runs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed42/training_summary.json)
- [测试指标](../runs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed42/test_metrics.json)

### M4-P

- [配置](../configs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet.yaml)
- [训练总结](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42/training_summary.json)
- [测试指标](../runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42/test_metrics.json)

### 数据划分

- [Clean manifest](../splits/kaggle_3m_multimodal_only_seed42.csv)
- [Clean split 元数据](../splits/kaggle_3m_multimodal_only_seed42.meta.json)
