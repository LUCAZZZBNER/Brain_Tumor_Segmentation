[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Device = "auto",
    [int[]]$Seeds = @(42, 123, 2026),
    [ValidateSet("All", "Configs", "Train", "Test", "Report")]
    [string]$Stage = "All"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildScript = Join-Path $PSScriptRoot "Build-Kaggle3MMultimodalOnlySplit.ps1"
$reportPath = (
    "reports/kaggle_3m_multimodal_only_complete_ablation_seeds42_123_2026.md"
)

$models = @(
    [pscustomobject]@{
        Name = "E0"
        BaseConfig = "configs/kaggle_3m_multimodal_only_e0_flair_unet.yaml"
        RunStem = "kaggle_3m_multimodal_only_e0_flair_unet"
    }
    [pscustomobject]@{
        Name = "E1-A"
        BaseConfig = "configs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation.yaml"
        RunStem = "kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation"
    }
    [pscustomobject]@{
        Name = "E2-B"
        BaseConfig = "configs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation.yaml"
        RunStem = "kaggle_3m_multimodal_only_e2_flair_unet_augmentation"
    }
    [pscustomobject]@{
        Name = "M0-AB"
        BaseConfig = "configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml"
        RunStem = "kaggle_3m_multimodal_only_m0_rgb_unet"
    }
    [pscustomobject]@{
        Name = "M4-NP"
        BaseConfig = (
            "configs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet.yaml"
        )
        RunStem = "kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet"
    }
    [pscustomobject]@{
        Name = "M4-P"
        BaseConfig = "configs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet.yaml"
        RunStem = "kaggle_3m_multimodal_only_m4_rgb_resnet34_unet"
    }
)

function Invoke-CheckedPython {
    param([string[]]$Arguments)

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Confirm-Python {
    try {
        & $Python --version
    }
    catch {
        throw "Python executable is unavailable: $Python. Activate the project environment or pass -Python <path>."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python executable failed with exit code $LASTEXITCODE`: $Python"
    }
}

function Get-ConfigPath($Model, [int]$Seed) {
    if ($Seed -eq 42) {
        return $Model.BaseConfig
    }
    $baseLeaf = [IO.Path]::GetFileNameWithoutExtension($Model.BaseConfig)
    return "configs/${baseLeaf}_seed${Seed}.yaml"
}

function Write-SeedConfig($Model, [int]$Seed) {
    if ($Seed -eq 42) {
        return
    }
    $basePath = Join-Path $projectRoot $Model.BaseConfig
    if (-not (Test-Path -LiteralPath $basePath -PathType Leaf)) {
        throw "Base config not found: $basePath"
    }
    $text = [IO.File]::ReadAllText($basePath, [Text.Encoding]::UTF8)
    $seedPattern = "(?m)^  seed: 42\r?$"
    if ([regex]::Matches($text, $seedPattern).Count -ne 1) {
        throw "Expected exactly one project seed in $basePath"
    }
    $outputPattern = "(?m)^  output_dir: (.+)_seed42\r?$"
    if ([regex]::Matches($text, $outputPattern).Count -ne 1) {
        throw "Expected exactly one seed42 output_dir in $basePath"
    }
    $derived = [regex]::Replace($text, $seedPattern, "  seed: $Seed")
    $derived = [regex]::Replace(
        $derived,
        $outputPattern,
        "  output_dir: `${1}_seed$Seed"
    )
    $configRelative = Get-ConfigPath $Model $Seed
    $configPath = Join-Path $projectRoot $configRelative
    [IO.File]::WriteAllText($configPath, $derived, [Text.UTF8Encoding]::new($false))
    Write-Host "Generated $configRelative" -ForegroundColor DarkCyan
}

function Confirm-CleanSplit {
    Write-Host "Rebuilding deterministic multimodal-only split..." -ForegroundColor Cyan
    & $buildScript
}

function Confirm-Seeds {
    if ($Seeds.Count -lt 1) {
        throw "At least one seed is required"
    }
    if (@($Seeds | Where-Object { $_ -le 0 }).Count -gt 0) {
        throw "Every seed must be a positive integer"
    }
    $script:Seeds = @($Seeds | Select-Object -Unique)
}

Push-Location $projectRoot
try {
    Confirm-Seeds
    Confirm-CleanSplit
    foreach ($seed in $Seeds) {
        foreach ($model in $models) {
            Write-SeedConfig $model $seed
        }
    }

    if ($Stage -ne "Configs") {
        Confirm-Python
    }

    if ($Stage -in @("All", "Train")) {
        foreach ($seed in $Seeds) {
            foreach ($model in $models) {
                $config = Get-ConfigPath $model $seed
                $runDir = Join-Path $projectRoot "runs\$($model.RunStem)_seed$seed"
                $summaryPath = Join-Path $runDir "training_summary.json"
                $bestCheckpoint = Join-Path $runDir "checkpoints\best.pt"
                $lastCheckpoint = Join-Path $runDir "checkpoints\last.pt"
                $metricsPath = Join-Path $runDir "metrics.jsonl"

                if ((Test-Path -LiteralPath $summaryPath -PathType Leaf) -and
                    (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
                    Write-Host "Skipping completed training: $($model.Name) seed=$seed" `
                        -ForegroundColor DarkYellow
                    continue
                }
                if (Test-Path -LiteralPath $metricsPath -PathType Leaf) {
                    if (-not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
                        throw "Incomplete run has no last checkpoint: $runDir"
                    }
                    Write-Host "Resuming $($model.Name) seed=$seed" -ForegroundColor Yellow
                    Invoke-CheckedPython -Arguments @(
                        "-m", "brain_tumor_seg.train",
                        "--config", $config,
                        "--device", $Device,
                        "--resume", $lastCheckpoint
                    )
                    continue
                }

                Write-Host "Training $($model.Name) seed=$seed" -ForegroundColor Cyan
                Invoke-CheckedPython -Arguments @(
                    "-m", "brain_tumor_seg.train",
                    "--config", $config,
                    "--device", $Device
                )
            }
        }
    }

    if ($Stage -in @("All", "Test")) {
        foreach ($seed in $Seeds) {
            foreach ($model in $models) {
                $config = Get-ConfigPath $model $seed
                $runDir = Join-Path $projectRoot "runs\$($model.RunStem)_seed$seed"
                $bestCheckpoint = Join-Path $runDir "checkpoints\best.pt"
                $testMetrics = Join-Path $runDir "test_metrics.json"
                if (Test-Path -LiteralPath $testMetrics -PathType Leaf) {
                    Write-Host "Skipping completed test: $($model.Name) seed=$seed" `
                        -ForegroundColor DarkYellow
                    continue
                }
                if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
                    throw "Best checkpoint not found: $bestCheckpoint"
                }

                Write-Host "Testing $($model.Name) seed=$seed" -ForegroundColor Green
                Invoke-CheckedPython -Arguments @(
                    "-m", "brain_tumor_seg.evaluate",
                    "--config", $config,
                    "--split", "test",
                    "--device", $Device
                )
            }
        }
    }

    if ($Stage -in @("All", "Report")) {
        $reportArguments = @(
            "scripts/summarize_kaggle3m_clean_six_model_multiseed.py",
            "--output", $reportPath,
            "--seeds"
        )
        $reportArguments += @($Seeds | ForEach-Object { [string]$_ })
        Write-Host "Generating six-model multi-seed report..." -ForegroundColor Cyan
        Invoke-CheckedPython -Arguments $reportArguments
    }
}
finally {
    Pop-Location
}

if ($Stage -eq "All") {
    Write-Host "Completed six-model multi-seed training, testing, and report." `
        -ForegroundColor Green
    Write-Host "Report: $reportPath" -ForegroundColor Green
}
