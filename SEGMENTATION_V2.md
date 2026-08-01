# Segmentation V2 dataset

`DATASET/Segmentation_v2` is a reproducible, leakage-controlled derivative of the original
`DATASET/Segmentation` and the BRISC 2025 segmentation task. The raw datasets are never modified.

## Current build

- 6,706 validated image/mask pairs
- 512x512, single-channel PNG images
- masks contain exactly `{0, 255}`
- class directories and `enh_<id>.png` naming are compatible with the existing loader
- source/class-stratified 70/15/15 split: 4,689 train, 1,008 validation, 1,009 test
- exact and conservative perceptual duplicate groups do not cross splits

The split is similarity-group controlled, but it is not proven patient-level independent because
neither source exposes reliable patient identifiers.

## Artifacts

- `DATASET/Segmentation_v2_manifest.csv`: provenance and content hashes for retained samples
- `DATASET/Segmentation_v2_quarantine.csv`: excluded conflicts and redundant candidates
- `DATASET/Segmentation_v2_report.json`: cleaning policy, counts, and validation summary
- `splits/segmentation_v2_seed42.csv`: immutable training manifest
- `splits/segmentation_v2_seed42.meta.json`: split fingerprint and metadata
- `configs/segmentation_v2.yaml`: baseline training configuration

## Rebuild

From the repository root in Windows PowerShell:

```powershell
& "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\Build-SegmentationV2.ps1 -Overwrite
```

`-Overwrite` deletes only the derived V2 artifacts named by the script. It does not modify either
raw dataset.

The safe cleaning policy quarantines:

- exact or perceptual duplicates with conflicting tumor classes;
- duplicate images whose normalized masks have IoU below 0.95;
- post-normalization duplicates with conflicting masks.

Same-class duplicates with mask IoU of at least 0.95 retain one canonical sample. During split
generation, remaining high-similarity same-source pairs are assigned to one split as a group.

## Train and evaluate

```powershell
python -m brain_tumor_seg.train --config configs/segmentation_v2.yaml
python -m brain_tumor_seg.evaluate --config configs/segmentation_v2.yaml
```

Do not run `make_splits` for this build unless intentionally replacing its immutable split. Rebuild
the dataset instead so cleaning, provenance, similarity grouping, and split assignment stay aligned.
