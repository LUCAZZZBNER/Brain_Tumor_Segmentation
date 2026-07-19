# 脑肿瘤 MRI 分割：可复现 U-Net Baseline

这是一个面向科研实验的二维脑肿瘤二值分割 baseline。项目将数据审计、固定划分、训练、模型选择、最终测试和单图推理分开，核心目标是得到一个可复现、可审计的前景 IoU 基准，而不是追求复杂模型或未经验证的高分。

> 当前仓库只包含实现和本地数据，不包含已经跑出的 IoU。代码完成后需要在目标训练环境中运行；请不要把验证集最佳 IoU 当成测试集结果。

## 1. 当前数据集结论

数据根目录为 `DATASET/Segmentation`，每张增强 MRI 灰度图和对应掩膜位于同一肿瘤类别目录：

```text
DATASET/Segmentation/
├── Glioma/
│   ├── enh_1841.png
│   └── enh_1841_mask.png
├── Meningioma/
└── Pituitary tumor/
```

全量审计结果：

| 肿瘤类别 | 图像/掩膜对 | 比例 |
|---|---:|---:|
| Glioma（胶质瘤） | 554 | 25.27% |
| Meningioma（脑膜瘤） | 708 | 32.30% |
| Pituitary tumor（垂体瘤） | 930 | 42.43% |
| 合计 | 2,192 | 100% |

- 原图和掩膜均为 `512×512`、单通道 `uint8` PNG；
- 2,192 对文件全部配对，没有损坏文件或完全相同的重复原图；
- 所有样本都包含肿瘤，没有正常脑部阴性样本；
- 掩膜的原始取值是 `{3, 255}`，不是常见的 `{0, 255}`；本项目统一使用 `mask >= 128` 得到 `{0, 1}`；
- 数据没有患者 ID、扫描方向、层号或其他 DICOM/NIfTI 元数据；
- 它很像由 Cheng 等人的 3,064 张增强 T1 MRI 数据转换而来。常见完整版本包含 1,426 张胶质瘤图像，而当前胶质瘤只有 554 张。编号中缺少 872 张，因此当前数据很可能是不完整子集。

严禁用 `mask > 0` 构造标签，因为背景值 3 也大于 0。掩膜缩放必须使用最近邻插值。

## 2. 项目结构

```text
.
├── configs/
│   └── baseline.yaml                 # 单一实验配置入口
├── splits/
│   ├── baseline_seed42.csv           # 固定样本清单、分组、集合和文件哈希
│   └── baseline_seed42.meta.json      # 比例、计数和清单哈希
├── src/brain_tumor_seg/
│   ├── config.py                     # 配置读取和约束检查
│   ├── splits.py                     # 数据审计、分层/分组划分、泄漏检查
│   ├── make_splits.py                # 生成一次性固定划分
│   ├── data.py                       # Dataset、成对增强、掩膜二值化
│   ├── model.py                      # 2D U-Net
│   ├── losses.py                     # BCE + soft Dice
│   ├── metrics.py                    # 宏/微 IoU 与 Dice
│   ├── engine.py                     # 训练和无梯度评估循环
│   ├── train.py                      # 只使用 train/val
│   ├── evaluate.py                   # 独立 val/test 评估
│   ├── predict.py                    # 未标注图像推理
│   └── utils.py                      # 随机种子、设备、日志和原子 checkpoint
├── tests/                            # 划分、掩膜、指标、模型和损失测试
├── pyproject.toml
└── requirements.txt
```

实验运行后默认写入：

```text
runs/unet_baseline_seed42/
├── resolved_config.json
├── environment.json
├── split_metadata_snapshot.json
├── metrics.jsonl
├── training_summary.json
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── test_metrics.json
└── predictions/test/
```

`runs/`、本地数据集和推理输出均已加入 `.gitignore`，固定划分清单应当纳入版本控制。

## 3. 严格划分协议

随机种子 42 下已生成并持久化如下划分：

| 类别 | Train | Validation | Test | 合计 |
|---|---:|---:|---:|---:|
| Glioma | 388 | 83 | 83 | 554 |
| Meningioma | 495 | 106 | 107 | 708 |
| Pituitary tumor | 650 | 140 | 140 | 930 |
| 合计 | 1,533 | 329 | 330 | 2,192 |

协议约束如下：

1. `train` 用于更新权重；
2. `val` 用于学习率调度、早停和选择 `best.pt`；
3. `test` 不会在 `train.py` 中创建 DataLoader，也不参与阈值、超参数或 epoch 选择；
4. 最终设计冻结后，使用 `evaluate.py --split test` 评估一次；
5. CSV 中保存每个原图和掩膜的 SHA-256，文件在第一次被对应流程使用前默认校验；元数据还保存清单本身的 SHA-256，checkpoint 同样记录清单哈希；
6. 程序会检查样本重复、分组跨集合、清单被修改、checkpoint 与清单不一致等问题；
7. 已有划分默认拒绝覆盖，避免反复重划测试集后挑选最好结果。

### 必须说明的患者级限制

当前 PNG 中没有患者 ID，因此已提交的划分只能保证**严格的样本级互斥**，不能证明患者级互斥。连续 MRI 切片可能来自同一患者；如果同一患者的切片进入不同集合，测试结果会偏乐观。在论文或报告中必须把它写成 sample-level split，不能声称 patient-independent evaluation。

若能取得原始 `.mat`、DICOM、NIfTI 或患者映射，创建完整 CSV：

```csv
sample_id,group_id
Glioma__enh_209,patient_001
Glioma__enh_210,patient_001
Meningioma__enh_1,patient_002
```

CSV 必须精确覆盖 2,192 个 `sample_id`，不多不少。然后：

1. 将 `data.group_csv` 指向该 CSV；
2. 为新实验修改 `data.manifest`、`data.split_metadata` 和 `project.output_dir`；
3. 重新运行 `make_splits`。

划分器会把同一 `group_id` 的所有切片放入同一集合，并在肿瘤类别内尽量逼近目标比例。患者级划分应优先于当前样本级 baseline。

## 4. 环境安装

推荐 Python 3.10–3.12。先根据本机 CUDA 版本从 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装合适的 PyTorch；随后在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

开发和测试依赖：

```powershell
python -m pip install -r requirements-dev.txt
```

确认 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 5. 复现实验

### 5.1 验证或重新生成固定划分

仓库已经包含当前数据对应的固定清单，正常复现**不需要重建**。如果换了数据或取得患者映射，再运行：

```powershell
python -m brain_tumor_seg.make_splits --config configs/baseline.yaml
```

如果目标清单已经存在，程序会停止。只有确定要定义一个全新实验协议时才使用 `--overwrite`；更推荐修改输出文件名并保留旧清单。

`make_splits` 会执行全量配对、图像模式、尺寸、掩膜值和非空前景检查，并计算内容哈希。`--skip-content-hashes` 可以加快生成，但不建议用于正式实验。

### 5.2 运行测试

```powershell
python -m pytest
```

### 5.3 训练

```powershell
python -m brain_tumor_seg.train --config configs/baseline.yaml --device auto
```

默认使用可用的 CUDA，否则回退至 Apple MPS 或 CPU。训练中断后可恢复：

```powershell
python -m brain_tumor_seg.train --config configs/baseline.yaml --resume runs/unet_baseline_seed42/checkpoints/last.pt
```

已有 `metrics.jsonl` 时，非恢复训练会拒绝覆盖。开始新实验时应修改 `project.output_dir`，保留旧实验作为完整记录。

### 5.4 验证最佳 checkpoint

这一步可以多次执行，用于确认保存模型的验证表现：

```powershell
python -m brain_tumor_seg.evaluate --config configs/baseline.yaml --split val
```

### 5.5 最终测试

模型、损失、增强、阈值和所有超参数冻结后执行：

```powershell
python -m brain_tumor_seg.evaluate --config configs/baseline.yaml --split test
```

结果写入 `runs/unet_baseline_seed42/test_metrics.json`。已有结果默认拒绝覆盖；只有在有明确审计理由、确实是在复现同一次冻结评估时才使用 `--overwrite-results`。默认最多保存 100 张二值预测掩膜；若只需要数值：

```powershell
python -m brain_tumor_seg.evaluate --config configs/baseline.yaml --split test --no-save-predictions
```

不要根据 test IoU 再调整模型；如需进一步开发，应把 test 重新封存，所有选择继续只在 train/val 上完成。

### 5.6 单图推理

```powershell
python -m brain_tumor_seg.predict DATASET/Segmentation/Glioma/enh_1841.png --config configs/baseline.yaml --output-dir predictions
```

输出包括二值掩膜 `*_mask.png` 和 8-bit 概率图 `*_probability.png`，最终尺寸恢复到输入图尺寸。

## 6. Baseline 方法

### 输入与预处理

- 输入：单通道 MRI；
- 原始尺寸：`512×512`；训练尺寸：`256×256`；
- 图像：双线性缩放，除以 255 后使用固定 `mean=0.5, std=0.5` 标准化；
- 掩膜：最近邻缩放，然后按 `>=128` 二值化；
- 训练增强：左右翻转、小角度旋转、轻微亮度和对比度变化；
- 验证和测试：不做随机增强。

固定标准化参数不依赖 val/test 统计量。若以后计算数据集均值和标准差，只能从 train 子集估计并记录到配置。

### 网络

模型为四层编码器/解码器 2D U-Net：

- 输入/输出通道：1/1；
- 初始通道数：32，逐层翻倍至 512；
- 每层使用两次 `Conv-BatchNorm-ReLU`；
- 转置卷积上采样和同尺度 skip connection；
- 中高层使用 `Dropout2d=0.1`；
- 输出是 raw logits，模型内部不做 sigmoid。

### 损失与优化

默认目标函数：

```text
Loss = 0.5 × BCEWithLogits + 0.5 × SoftDiceLoss
```

Dice 项缓解肿瘤前景仅占约 1%–2% 带来的类别不平衡。优化器为 AdamW，初始学习率 `1e-3`，weight decay `1e-4`；验证 IoU 停滞时学习率减半，15 个 epoch 无提升则早停。CUDA 环境默认启用 AMP，梯度范数裁剪为 1.0。

## 7. IoU 定义与模型选择

对阈值 0.5 后的前景集合 `P` 和真实前景 `G`：

```text
IoU = |P ∩ G| / |P ∪ G|
```

主指标 `macro_iou` 是先对每张图计算前景 IoU，再对图像等权平均。这样大肿瘤不会仅因为像素多而完全支配结果。项目同时报告：

- `micro_iou`：在整个集合累加交集与并集后计算；
- `macro_dice`：逐图 Dice 的平均；
- `micro_dice`：全集合像素级 Dice；
- `per_class`：分别报告三种肿瘤类别的上述指标。

模型只按 `val/macro_iou` 选择 `best.pt`。当前数据没有空掩膜；指标实现仍规定预测和真值都为空时该图 IoU/Dice 为 1。

## 8. 配置说明

所有实验参数位于 `configs/baseline.yaml`：

- `project.seed`：Python、NumPy、PyTorch 和 DataLoader 随机种子；
- `data.*`：路径、固定划分、尺寸、batch、worker 和增强；
- `model.*`：U-Net 通道和 dropout；
- `loss.*`：BCE/Dice 权重；
- `optimizer.*`、`scheduler.*`：优化策略；
- `training.deterministic=true`：请求确定性算法，可能牺牲少量速度；
- `metrics.threshold`：固定为 0.5，不应在 test 上搜索；
- `evaluation.*`：测试集合与预测保存数量。

如果显存不足，优先减小 `data.batch_size`；若仍不足，可把 `model.base_channels` 从 32 改为 16。任何改变都应使用新的 `project.output_dir` 并记录成独立实验。

## 9. 科研报告建议

至少报告以下内容：

1. 数据来源、当前缺失的胶质瘤样本和仅含阳性病例的限制；
2. 是 sample-level 还是 patient-level split，以及精确的集合数量；
3. 随机种子、清单哈希、输入尺寸、增强、损失、优化器和训练轮数；
4. checkpoint 选择只使用验证集；
5. test 的 macro/micro IoU、Dice、分类别结果和定性预测；
6. 至少 3–5 个不同随机种子的重复实验，报告均值和标准差；
7. 失败案例，尤其是微小肿瘤、边界模糊和模型假阳性；
8. 当前结果不能代表对正常人群、三维体数据或外部医院数据的泛化。

更正式的后续实验应优先补全 872 张疑似缺失胶质瘤图像、恢复患者 ID、加入无肿瘤阴性病例并增加外部测试集。

## 10. 常见问题

### 为什么不直接把 RGB 作为三通道输入？

本地文件实际是单通道灰度 PNG。复制到三通道不会增加信息，只会增加第一层参数量。

### 为什么训练脚本不直接输出 test IoU？

为了隔离测试集。每个 epoch 都查看 test 会把测试集变成另一个验证集，使最终结果产生选择偏差。

### 为什么不采用多类别分割？

每张图只有一种肿瘤且掩膜本身只有背景/肿瘤。当前 baseline 的目标是定位任意肿瘤。文件夹类别只用于分层划分和分类别评估；若要做四类像素分割，需要单独定义标签编码和研究问题。

### 为什么无法保证患者级独立？

PNG 转换结果没有保留患者标识，文件编号不能可靠等价于患者。用连续编号或图像相似度猜患者都会引入不可验证假设；可靠方案是从原始数据恢复 patient ID 并通过 `group_csv` 重新划分。

## 参考

- Ronneberger O, Fischer P, Brox T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI, 2015.
- Cheng J, et al. *Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition*. PLOS ONE, 2015. 数据来源仍应根据你实际下载页面和许可证再次核实后再写入论文。
