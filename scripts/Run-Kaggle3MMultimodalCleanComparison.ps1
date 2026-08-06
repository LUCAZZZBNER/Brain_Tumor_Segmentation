[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Device = "auto",
    [ValidateSet("All", "Split", "Train", "Test", "Report")]
    [string]$Stage = "All"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildScript = Join-Path $PSScriptRoot "Build-Kaggle3MMultimodalOnlySplit.ps1"
$manifestPath = Join-Path $projectRoot "splits\kaggle_3m_multimodal_only_seed42.csv"
$metadataPath = Join-Path $projectRoot "splits\kaggle_3m_multimodal_only_seed42.meta.json"
$reportPath = "reports/kaggle_3m_multimodal_only_e0_m0_m4.md"

$experiments = @(
    [pscustomobject]@{
        Name = "E0"
        Config = "configs/kaggle_3m_multimodal_only_e0_flair_unet.yaml"
        Output = "runs/kaggle_3m_multimodal_only_e0_flair_unet_seed42"
    }
    [pscustomobject]@{
        Name = "M0"
        Config = "configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml"
        Output = "runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42"
    }
    [pscustomobject]@{
        Name = "M4"
        Config = "configs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet.yaml"
        Output = "runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42"
    }
)

function Invoke-CheckedPython {
    param([string[]]$Arguments)

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Build-CleanSplit {
    Write-Host "Building multimodal-only split..." -ForegroundColor Cyan
    & $buildScript
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

Push-Location $projectRoot
try {
    if ($Stage -ne "Split") {
        Confirm-Python
    }
    if ($Stage -in @("All", "Split", "Train")) {
        Build-CleanSplit
    }
    elseif (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf) -or
            -not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        Build-CleanSplit
    }

    if ($Stage -in @("All", "Train")) {
        foreach ($experiment in $experiments) {
            $runDir = Join-Path $projectRoot $experiment.Output
            $summaryPath = Join-Path $runDir "training_summary.json"
            $checkpointPath = Join-Path $runDir "checkpoints\best.pt"
            $lastCheckpointPath = Join-Path $runDir "checkpoints\last.pt"
            $metricsPath = Join-Path $runDir "metrics.jsonl"
            if ((Test-Path -LiteralPath $summaryPath -PathType Leaf) -and
                (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
                Write-Host "Skipping completed training: $($experiment.Name)" -ForegroundColor DarkYellow
                continue
            }
            if (Test-Path -LiteralPath $metricsPath -PathType Leaf) {
                if (-not (Test-Path -LiteralPath $lastCheckpointPath -PathType Leaf)) {
                    throw "Incomplete run has no last checkpoint: $runDir"
                }
                Write-Host "Resuming interrupted training: $($experiment.Name)" -ForegroundColor Yellow
                Invoke-CheckedPython -Arguments @(
                    "-m", "brain_tumor_seg.train",
                    "--config", $experiment.Config,
                    "--device", $Device,
                    "--resume", $lastCheckpointPath
                )
                continue
            }

            Write-Host "Training $($experiment.Name): $($experiment.Config)" -ForegroundColor Cyan
            Invoke-CheckedPython -Arguments @(
                "-m", "brain_tumor_seg.train",
                "--config", $experiment.Config,
                "--device", $Device
            )
        }
    }

    if ($Stage -in @("All", "Test")) {
        foreach ($experiment in $experiments) {
            $runDir = Join-Path $projectRoot $experiment.Output
            $checkpointPath = Join-Path $runDir "checkpoints\best.pt"
            $testMetricsPath = Join-Path $runDir "test_metrics.json"
            if (Test-Path -LiteralPath $testMetricsPath -PathType Leaf) {
                Write-Host "Skipping completed test: $($experiment.Name)" -ForegroundColor DarkYellow
                continue
            }
            if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
                throw "Best checkpoint not found for $($experiment.Name): $checkpointPath"
            }

            Write-Host "Testing $($experiment.Name) on the frozen test split..." -ForegroundColor Green
            Invoke-CheckedPython -Arguments @(
                "-m", "brain_tumor_seg.evaluate",
                "--config", $experiment.Config,
                "--split", "test",
                "--device", $Device
            )
        }
    }

    if ($Stage -in @("All", "Report")) {
        Write-Host "Generating Markdown comparison report..." -ForegroundColor Cyan
        Invoke-CheckedPython -Arguments @(
            "scripts/summarize_kaggle3m_multimodal_clean.py",
            "--output", $reportPath
        )
    }
}
finally {
    Pop-Location
}

if ($Stage -eq "All") {
    Write-Host "Completed split, E0/M0/M4 training, test evaluation, and report." -ForegroundColor Green
    Write-Host "Report: $reportPath" -ForegroundColor Green
}
