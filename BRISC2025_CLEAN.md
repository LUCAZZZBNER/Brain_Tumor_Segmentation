# BRISC2025 clean segmentation protocol

This dataset contains only BRISC2025 segmentation samples retained by the conservative
Segmentation_v2 cleaning pass. It does not contain samples from the original Segmentation dataset.

The official BRISC test split is frozen. Validation is selected only from the official training
split, and perceptually similar samples are assigned together. Any official-training sample linked
to an official-test similarity group is quarantined instead of used for training or validation.

This controls exact/perceptual sample leakage and prevents official-test reuse during training.
BRISC does not provide reliable patient identifiers, so patient-level independence cannot be
proven and must not be claimed.

Build from the repository root:

```powershell
& "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\Build-BRISC2025Clean.ps1
```

Train the no-augmentation U-Net baseline:

```powershell
python -m brain_tumor_seg.train `
  --config configs/brisc2025_clean_unet_no_augmentation.yaml `
  --device auto
```

Evaluate the frozen official test split using the best validation checkpoint:

```powershell
python -m brain_tumor_seg.evaluate `
  --config configs/brisc2025_clean_unet_no_augmentation.yaml `
  --split test `
  --device auto
```
