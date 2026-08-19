# Seed-42 modern 2D baselines

This workflow evaluates two unenhanced, no-augmentation baselines on the existing fixed
`kaggle_3m_multimodal_only_seed42` patient manifest.

## Experiment identity

- Source cohort: 104 cleaned patients and 3,629 slices.
- Fixed patients: train 85, validation 9, test 10.
- Fixed slices: train 2,619, validation 485, test 525.
- Random seed: 42.
- Input: the three stored MRI channels at 256 x 256.
- Test threshold: 0.5.
- No geometric or intensity data augmentation.
- Test patients are not read during training or validation.
- Windows-safe nnU-Net resources: batch size 4, data workers 0, and
  preprocessing/export workers 1. These are resource-only overrides; they do not change the
  model, patient split, seed, loss, or no-augmentation policy.
- Windows-safe TransUNet resources: batch size 2 and data workers 0.

## Baselines

### Official nnU-Net v2 2D

The nnU-Net workflow uses the official `nnunetv2` planner, preprocessing, network, and
training pipeline. `nnUNetTrainerNoDA` disables data augmentation. The repository converter
writes an explicit `splits_final.json`, so nnU-Net cannot replace the fixed patient split with
its own cross-validation split. The best validation checkpoint is used for the frozen test set,
and inference-time mirroring/TTA is disabled.

Install the optional official dependency in the environment used for training:

```powershell
python -m pip install -r requirements-nnunet.txt
```

### Basic TransUNet 2D

The repository-native baseline is a fixed CNN encoder, four-layer Transformer bottleneck,
and U-Net decoder. It is randomly initialized and uses no architecture-specific pretraining,
auxiliary head, or data augmentation. It is a controlled basic TransUNet implementation, not
the pretrained R50-ViT-B/16 configuration from the original TransUNet paper; this distinction
must be stated in a manuscript comparison table.

## One-command training and testing

Run both baselines:

```powershell
& .\scripts\Run-ModernBaselinesSeed42.ps1 -Python python -Baseline Both -Stage All -Device cuda
```

Run only one baseline:

```powershell
& .\scripts\Run-ModernBaselinesSeed42.ps1 -Python python -Baseline NnUNet -Stage All -Device cuda
& .\scripts\Run-ModernBaselinesSeed42.ps1 -Python python -Baseline TransUNet -Stage All -Device cuda
```

If `python` is not on `PATH`, pass the active environment executable explicitly:

```powershell
& .\scripts\Run-ModernBaselinesSeed42.ps1 `
  -Python D:\path\to\venv\Scripts\python.exe `
  -Baseline Both -Stage All -Device cuda
```

The script skips completed stages and resumes incomplete training when a latest checkpoint is
available. If no latest checkpoint exists after an interrupted epoch, training safely starts
again while reusing the converted and preprocessed data. Individual stages can be selected with
`-Stage Prepare`, `SmokeTest`, `Train`, `Test`, or `Report`.

Before a long run, execute one real forward/backward batch without touching the actual results
folder:

```powershell
& .\scripts\Run-ModernBaselinesSeed42.ps1 `
  -Python D:\Dev\Miniconda3\envs\medical_cv\python.exe `
  -Baseline NnUNet -Stage SmokeTest -Device cuda
```

## Result files

- nnU-Net: `runs/nnunetv2_2d_kaggle_3m_clean_no_augmentation_seed42/test_metrics.json`
- TransUNet: `runs/kaggle_3m_multimodal_only_transunet_2d_basic_no_augmentation_seed42/test_metrics.json`
- Combined Markdown report: `reports/kaggle_3m_modern_baselines_seed42.md`

Both results include the fixed manifest hash and the same project metrics, including Positive
Macro IoU, Positive Dice, Micro IoU, Precision, Recall, and empty-slice false-positive rate.
`-Stage All` generates the combined report automatically; `-Stage Report` can regenerate it at
any time and marks unfinished models as pending instead of failing.
