# Kaggle 3M 三通道模型结构消融实验报告（M0–M4）

## 1. 实验目的

本实验以三通道 RGB 多模态 U-Net 为参考模型，比较 UNet++、Attention U-Net、ASPP U-Net 和预训练 ResNet34 U-Net 的训练收敛、验证集表现和测试集分割性能，重点分析不同模型对微小病灶、中大病灶和空切片误报的影响。

实验日期：2026-08-03 至 2026-08-04。

## 2. 实验设置

五组实验均使用：

- Kaggle 3M 数据集
- patient-level 固定划分，随机种子 42
- 三通道多模态输入
- 256×256 输入分辨率
- batch size 4
- AdamW，学习率 `3e-4`
- BCE + Positive Dice loss
- 正切片采样比例 0.40
- 水平翻转与 ±5° 旋转
- 最多训练 150 epochs，早停耐心 30
- 以 Validation Positive Macro IoU 选择最佳 checkpoint
- 固定二值化阈值 0.5

模型定义：

| 编号 | 模型 | 关键配置 |
|---|---|---|
| M0 | RGB U-Net | base channels 32，无 BatchNorm，无 dropout |
| M1 | RGB UNet++ | base channels 32，无 BatchNorm，无 dropout |
| M2 | RGB Attention U-Net | base channels 32，无 BatchNorm，无 dropout |
| M3 | RGB ASPP U-Net | ASPP rates `[1,2,4,8]`，BatchNorm，dropout 0.10 |
| M4 | RGB ResNet34 U-Net | ImageNet 预训练 ResNet34 encoder |

注意：M3 原始无 BatchNorm 版本在第 3 个 epoch 出现 non-finite loss，因此最终使用带 BatchNorm 和 dropout 的稳定版本。M4 同时包含结构变化和 ImageNet 预训练因素，不能把其提升完全归因于 ResNet34 结构。

## 3. 训练结果

| 模型 | 最佳 Epoch | 最后 Epoch | 训练时间 | 最佳点 Train Positive IoU | 最佳点 Val Positive IoU | Train–Val Gap |
|---|---:|---:|---:|---:|---:|---:|
| M0 U-Net | 45 | 75 | 32.1 min | 0.8056 | 0.6434 | 0.1622 |
| M1 UNet++ | 43 | 73 | 64.0 min | 0.8070 | 0.6302 | 0.1768 |
| M2 Attention U-Net | 50 | 80 | 38.1 min | 0.8187 | 0.6074 | 0.2113 |
| M3 ASPP U-Net | 71 | 101 | 53.5 min | 0.8247 | 0.6440 | 0.1807 |
| M4 ResNet34 U-Net | 61 | 91 | 47.2 min | **0.8478** | **0.7037** | **0.1441** |

### 3.1 最后一个 epoch 的表现

| 模型 | Last Train Positive IoU | Last Val Positive IoU | 训练后期现象 |
|---|---:|---:|---|
| M0 U-Net | 0.8711 | 0.5303 | 明显过拟合 |
| M1 UNet++ | 0.8732 | 0.5608 | 明显过拟合 |
| M2 Attention U-Net | 0.8669 | 0.5151 | 过拟合最严重 |
| M3 ASPP U-Net | 0.8290 | 0.6217 | 后期下降相对较小 |
| M4 ResNet34 U-Net | 0.8725 | **0.6737** | 泛化保持最好 |

训练结论：

- M4 的最佳验证 IoU 最高，最佳点 Train–Val gap 最小，说明预训练 encoder 提升了泛化能力。
- M1 训练耗时约为 M0 的两倍，但验证 IoU低于 M0，计算成本没有转化为性能收益。
- M2 的 Train–Val gap 最大，Attention Gate 没有缓解当前数据上的过拟合。
- M3 收敛最慢，但训练后期验证性能相对稳定；其 BatchNorm 和 dropout 起到了正则化作用。

## 4. 测试集主要结果

`Positive Macro IoU` 只统计真实包含肿瘤的切片，是本报告的主要指标。`Micro IoU` 汇总全部 TP、FP、FN，更容易受到中大病灶影响。`Macro IoU` 包含空切片，容易被空切片数量和误报率影响，不应单独作为模型优劣结论。

| 模型 | Positive Macro IoU | Micro IoU | Macro IoU | Positive Macro Dice | Test Loss |
|---|---:|---:|---:|---:|---:|
| M0 U-Net | 0.7192 | 0.7669 | **0.8375** | 0.8029 | 0.0527 |
| M1 UNet++ | 0.7027 | 0.7380 | 0.8028 | 0.7878 | 0.0552 |
| M2 Attention U-Net | 0.6885 | 0.7234 | 0.8080 | 0.7775 | 0.0587 |
| M3 ASPP U-Net | 0.7022 | 0.7257 | 0.8173 | 0.7903 | 0.0549 |
| M4 ResNet34 U-Net | **0.7471** | **0.7784** | 0.8041 | **0.8261** | **0.0460** |

### 4.1 相对 M0 的变化

| 模型 | Positive IoU 变化 | Micro IoU 变化 | 结论 |
|---|---:|---:|---|
| M1 UNet++ | -0.0165 | -0.0289 | 未超过普通 U-Net |
| M2 Attention U-Net | -0.0307 | -0.0435 | 整体退步最大 |
| M3 ASPP U-Net | -0.0170 | -0.0412 | 整体退步，但微小病灶改善明显 |
| M4 ResNet34 U-Net | **+0.0279** | **+0.0115** | 唯一稳定超过 M0 的候选 |

测试集总体排名：

1. M4 ResNet34 U-Net：Positive IoU 0.7471
2. M0 普通 U-Net：Positive IoU 0.7192
3. M1 UNet++：Positive IoU 0.7027
4. M3 ASPP U-Net：Positive IoU 0.7022
5. M2 Attention U-Net：Positive IoU 0.6885

## 5. Precision、Recall 与空切片误报

| 模型 | Micro Precision | Micro Recall | 空切片误报率 | 空切片平均误报像素 |
|---|---:|---:|---:|---:|
| M0 U-Net | 0.9027 | 0.8360 | **10.60%** | 29.6 |
| M1 UNet++ | 0.8853 | 0.8160 | 14.94% | 50.9 |
| M2 Attention U-Net | 0.8687 | 0.8122 | 13.49% | 59.5 |
| M3 ASPP U-Net | **0.9177** | 0.7763 | 12.77% | **22.6** |
| M4 ResNet34 U-Net | 0.8760 | **0.8748** | 16.87% | 56.3 |

解释：

- M3 的 Precision 最高、平均误报面积最小，但 Recall 最低，说明 ASPP 模型更加保守，容易漏掉部分病灶。
- M4 的 Recall 最高，因此肿瘤覆盖更完整，但代价是空切片误报率升至 16.87%。
- M0 的空切片误报控制最好，并且 Precision 与 Recall 更均衡。
- M4 的 Macro IoU 低于 M0，主要是空切片误报较多，而不是正病灶分割更差。

空切片严重误报统计：

| 模型 | 有任意误报的空切片 | 误报 ≥100 px | 误报 ≥1000 px | 最大误报像素 |
|---|---:|---:|---:|---:|
| M0 U-Net | 44/415 | 18 | 3 | 2792 |
| M1 UNet++ | 62/415 | 32 | 4 | 5289 |
| M2 Attention U-Net | 56/415 | 32 | 7 | 4380 |
| M3 ASPP U-Net | 53/415 | 25 | **1** | **1145** |
| M4 ResNet34 U-Net | 70/415 | 47 | 6 | 4113 |

## 6. 不同病灶大小的 IoU

| 模型 | 微小 1–255 px | 小型 256–1023 px | 中型 1024–4095 px | 大型 ≥4096 px |
|---|---:|---:|---:|---:|
| M0 U-Net | 0.3585 | 0.6194 | 0.7836 | 0.8641 |
| M1 UNet++ | 0.3431 | 0.6211 | 0.7546 | **0.8771** |
| M2 Attention U-Net | 0.2976 | 0.6068 | 0.7452 | 0.8601 |
| M3 ASPP U-Net | **0.5195** | 0.6341 | 0.7358 | 0.8097 |
| M4 ResNet34 U-Net | 0.3668 | **0.6359** | **0.8238** | 0.8628 |

病灶分层结论：

- M3 对微小病灶最强，IoU 由 M0 的 0.3585 提升到 0.5195，绝对提升 0.1610。
- M4 在小型和中型病灶上最佳，是其总体指标提升的主要来源。
- M1 仅在大型病灶上略高于其他模型，但整体收益不足以抵消计算成本。
- M2 在所有病灶尺度上均没有明显优势。
- 如果任务强调微小病灶检出，M3 仍具有独立研究价值；如果强调总体分割性能，M4 更合适。

## 7. 患者级结果与不确定性

| 模型 | Patient Macro Positive IoU | 最差患者 | 最佳患者 |
|---|---:|---:|---:|
| M0 U-Net | 0.7049 | 0.3867 | 0.8841 |
| M1 UNet++ | 0.6935 | 0.4067 | 0.8884 |
| M2 Attention U-Net | 0.6782 | 0.4012 | 0.8888 |
| M3 ASPP U-Net | 0.6900 | 0.3795 | 0.8895 |
| M4 ResNet34 U-Net | **0.7293** | **0.4481** | **0.9034** |

M4 相对 M0 的患者级平均提升为 0.0244。但测试集只有 11 名患者，粗略 95% 区间为 `[-0.0350, 0.0838]`，仍然跨过 0。因此 M4 是当前最有希望的模型，但还需要多个随机种子或 patient-level 交叉验证确认稳定性。

## 8. 各模型综合评价

### M0 RGB U-Net

- 综合表现第二。
- 训练速度最快。
- 空切片误报控制最好。
- Precision 和 Recall 较均衡。
- 仍是很强且可靠的结构基线。

### M1 RGB UNet++

- 训练耗时约为 M0 的两倍。
- Positive IoU、Micro IoU 和空切片表现均低于 M0。
- 只在大型病灶上有轻微优势。
- 当前配置下不建议继续作为主线。

### M2 RGB Attention U-Net

- Train–Val gap 最大。
- 总体 Positive IoU 最低。
- Attention Gate 没有减少空切片误报，也没有改善微小病灶。
- 不建议继续直接堆叠 Attention 模块。

### M3 RGB ASPP U-Net

- 微小病灶 IoU 显著最佳。
- Precision 最高，严重误报数量最少。
- Recall 偏低，中大病灶性能下降。
- 适合作为微小病灶专家模型或后续多尺度分支，不适合直接替换总体模型。

### M4 RGB ResNet34 U-Net

- Validation Positive IoU、Test Positive IoU、Micro IoU、Positive Dice 和 Test Loss 均为最佳。
- Train–Val gap 最小，泛化保持最好。
- 小型和中型病灶表现最佳。
- Recall 最高，但空切片误报较多。
- 当前总体最佳候选。

## 9. 最终结论

当前推荐主模型：

> M4：RGB 三通道 + 轻量增强 + ImageNet 预训练 ResNet34 U-Net。

其 Test Positive Macro IoU 为 0.7471，相比 M0 普通 U-Net 的 0.7192 提升 0.0279；Test Positive Macro Dice 从 0.8029 提升到 0.8261。

适合报告或论文的表述：

> 在相同三通道输入与训练策略下，预训练 ResNet34 U-Net 获得最佳总体分割性能，其 Test Positive Macro IoU 达到 0.7471，较普通 U-Net 提升 2.79 个百分点。ASPP U-Net 虽然总体 IoU 未超过普通 U-Net，但在微小病灶上取得 0.5195 的 IoU，相比普通 U-Net 提升 16.10 个百分点，说明多尺度上下文对微小病灶具有明显价值。UNet++ 与 Attention U-Net 在当前数据集和训练设置下均未取得有效提升。

下一步建议：

1. 以 M4 为主模型，优先解决空切片误报。
2. 尝试将轻量 ASPP 放入 ResNet34 bottleneck，但必须与 M4 做单变量消融。
3. 或对 M3 与 M4 进行验证集概率融合，观察能否同时保留 M3 的微小病灶能力和 M4 的总体性能。
4. 增加 tumor-presence 辅助分类头，重点降低 M4 的空切片误报。
5. 使用多个随机种子或 patient-level 交叉验证验证 M4 的稳定性。

## 10. 原始结果文件

- [M0 RGB U-Net 训练总结](../runs/kaggle_3m_e1_a_rgb_seed42/training_summary.json)
- [M0 RGB U-Net 测试结果](../runs/kaggle_3m_e1_a_rgb_seed42/test_metrics.json)
- [M1 RGB UNet++ 训练总结](../runs/kaggle_3m_m1_rgb_unet_plus_plus_seed42/training_summary.json)
- [M1 RGB UNet++ 测试结果](../runs/kaggle_3m_m1_rgb_unet_plus_plus_seed42/test_metrics.json)
- [M2 RGB Attention U-Net 训练总结](../runs/kaggle_3m_m2_rgb_attention_unet_seed42/training_summary.json)
- [M2 RGB Attention U-Net 测试结果](../runs/kaggle_3m_m2_rgb_attention_unet_seed42/test_metrics.json)
- [M3 RGB ASPP U-Net 训练总结](../runs/kaggle_3m_m3_rgb_aspp_unet_stable_seed42/training_summary.json)
- [M3 RGB ASPP U-Net 测试结果](../runs/kaggle_3m_m3_rgb_aspp_unet_stable_seed42/test_metrics.json)
- [M4 RGB ResNet34 U-Net 训练总结](../runs/kaggle_3m_m4_rgb_resnet34_unet_seed42/training_summary.json)
- [M4 RGB ResNet34 U-Net 测试结果](../runs/kaggle_3m_m4_rgb_resnet34_unet_seed42/test_metrics.json)

