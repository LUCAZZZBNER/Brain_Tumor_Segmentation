[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Device = "auto",
    [ValidateSet("All", "Train", "Test", "Report")]
    [string]$Stage = "All"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildScript = Join-Path $PSScriptRoot "Build-Kaggle3MMultimodalOnlySplit.ps1"
$reportPath = "reports/kaggle_3m_m4_pretraining_ablation.md"

$experiments = @(
    [pscustomobject]@{
        Name = "M0"
        Config = "configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml"
        Output = "runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42"
    }
    [pscustomobject]@{
        Name = "M4-NP"
        Config = "configs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet.yaml"
        Output = "runs/kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed42"
    }
    [pscustomobject]@{
        Name = "M4-P"
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

function Confirm-CleanSplit {
    Write-Host "Rebuilding deterministic multimodal-only split..." -ForegroundColor Cyan
    & $buildScript
}

Push-Location $projectRoot
try {
    Confirm-Python
    Confirm-CleanSplit

    if ($Stage -in @("All", "Train")) {
        foreach ($experiment in $experiments) {
            $runDir = Join-Path $projectRoot $experiment.Output
            $summaryPath = Join-Path $runDir "training_summary.json"
            $bestCheckpoint = Join-Path $runDir "checkpoints\best.pt"
            $lastCheckpoint = Join-Path $runDir "checkpoints\last.pt"
            $metricsPath = Join-Path $runDir "metrics.jsonl"

            if ((Test-Path -LiteralPath $summaryPath -PathType Leaf) -and
                (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
                Write-Host "Skipping completed training: $($experiment.Name)" -ForegroundColor DarkYellow
                continue
            }
            if (Test-Path -LiteralPath $metricsPath -PathType Leaf) {
                if (-not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
                    throw "Incomplete run has no last checkpoint: $runDir"
                }
                Write-Host "Resuming interrupted training: $($experiment.Name)" -ForegroundColor Yellow
                Invoke-CheckedPython -Arguments @(
                    "-m", "brain_tumor_seg.train",
                    "--config", $experiment.Config,
                    "--device", $Device,
                    "--resume", $lastCheckpoint
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
            $bestCheckpoint = Join-Path $runDir "checkpoints\best.pt"
            $testMetrics = Join-Path $runDir "test_metrics.json"
            if (Test-Path -LiteralPath $testMetrics -PathType Leaf) {
                Write-Host "Skipping completed test: $($experiment.Name)" -ForegroundColor DarkYellow
                continue
            }
            if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
                throw "Best checkpoint not found for $($experiment.Name): $bestCheckpoint"
            }

            Write-Host "Testing $($experiment.Name)..." -ForegroundColor Green
            Invoke-CheckedPython -Arguments @(
                "-m", "brain_tumor_seg.evaluate",
                "--config", $experiment.Config,
                "--split", "test",
                "--device", $Device
            )
        }
    }

    if ($Stage -in @("All", "Report")) {
        Write-Host "Generating M4 pretraining ablation report..." -ForegroundColor Cyan
        Invoke-CheckedPython -Arguments @(
            "scripts/summarize_kaggle3m_m4_pretraining_ablation.py",
            "--output", $reportPath
        )
    }
}
finally {
    Pop-Location
}

if ($Stage -eq "All") {
    Write-Host "Completed M0/M4-NP/M4-P training, testing, and report." -ForegroundColor Green
    Write-Host "Report: $reportPath" -ForegroundColor Green
}

