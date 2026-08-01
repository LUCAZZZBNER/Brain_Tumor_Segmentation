# 训练、验证和测试过程记录

训练命令保持不变：

```powershell
python -m brain_tumor_seg.train --config configs/segmentation_v2.yaml --device auto
```

每个 epoch 完成后都会更新以下产物：

```text
runs/<experiment>/
├── metrics.jsonl
├── history/
│   ├── epochs.csv
│   ├── batches/
│   │   ├── train_epoch_001.csv
│   │   ├── val_epoch_001.csv
│   │   └── ...
│   ├── validation_samples/
│   │   ├── epoch_001.csv
│   │   └── ...
│   └── curves/
│       ├── loss.png
│       ├── iou.png
│       ├── dice.png
│       ├── precision_recall.png
│       ├── specificity_accuracy.png
│       ├── model_selection.png
│       └── learning_rate.png
└── checkpoints/
    ├── best.pt
    └── last.pt
```

批次 CSV 保存 batch loss、累计 loss、IoU、Dice、Precision、Recall、Specificity 和
Accuracy。`validation_samples` 保存每个验证样本的混淆矩阵像素计数和逐样本指标。
`epochs.csv` 与曲线从追加式 `metrics.jsonl` 重建，因此恢复训练时会延续原历史。

最终测试仍通过独立命令执行，不会在每个训练 epoch 上读取 test：

```powershell
python -m brain_tumor_seg.evaluate --config configs/segmentation_v2.yaml --split test
```

测试额外生成：

```text
runs/<experiment>/
├── test_metrics.json
├── evaluation/test/
│   ├── batches.csv
│   ├── samples.csv
│   ├── summary.csv
│   ├── per_class.csv
│   └── plots/
│       ├── sample_metric_distributions.png
│       └── per_class_metrics.png
├── predictions/test/<tumor_type>/
│   ├── <sample>_pred.png
│   └── <sample>_prob.png
└── comparisons/test/<tumor_type>/
    └── <sample>.png
```

每张 comparison 图包含原始 MRI、真值掩膜、预测掩膜和叠加误差图。叠加图中绿色为
TP、红色为 FP、黄色为 FN。

配置项：

```yaml
evaluation:
  save_predictions: true
  save_probability_maps: true
  save_comparison_figures: true
  max_saved_predictions: null  # null 表示保存全部；整数表示最多保存多少张
```

`--no-save-predictions` 会关闭预测图、概率图和 comparison 图，但仍保存测试批次、逐样本
指标和汇总图。`--overwrite-results` 会明确删除并重建同一 split 的旧评估可视化目录。
