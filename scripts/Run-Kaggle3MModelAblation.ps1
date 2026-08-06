[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Device = "auto",
    [switch]$ValidationOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$split = if ($ValidationOnly) { "val" } else { "test" }
$experiments = @(
    [pscustomobject]@{
        Name = "M1"
        Config = "configs/kaggle_3m_m1_rgb_unet_plus_plus.yaml"
        Output = "runs/kaggle_3m_m1_rgb_unet_plus_plus_seed42"
    }
    [pscustomobject]@{
        Name = "M2"
        Config = "configs/kaggle_3m_m2_rgb_attention_unet.yaml"
        Output = "runs/kaggle_3m_m2_rgb_attention_unet_seed42"
    }
    [pscustomobject]@{
        Name = "M3"
        Config = "configs/kaggle_3m_m3_rgb_aspp_unet.yaml"
        Output = "runs/kaggle_3m_m3_rgb_aspp_unet_stable_seed42"
    }
    [pscustomobject]@{
        Name = "M4"
        Config = "configs/kaggle_3m_m4_rgb_resnet34_unet.yaml"
        Output = "runs/kaggle_3m_m4_rgb_resnet34_unet_seed42"
    }
)

Push-Location $projectRoot
try {
    foreach ($experiment in $experiments) {
        $config = $experiment.Config
        $runDir = Join-Path $projectRoot $experiment.Output
        $summaryPath = Join-Path $runDir "training_summary.json"
        $testMetricsPath = Join-Path $runDir "test_metrics.json"
        if (-not $ValidationOnly -and (Test-Path $summaryPath) -and (Test-Path $testMetricsPath)) {
            Write-Host "Skipping completed $($experiment.Name): $config" -ForegroundColor DarkYellow
            continue
        }

        Write-Host "Training $($experiment.Name): $config" -ForegroundColor Cyan
        & $Python -m brain_tumor_seg.train --config $config --device $Device
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $config with exit code $LASTEXITCODE"
        }

        Write-Host "Evaluating $config on $split" -ForegroundColor Green
        & $Python -m brain_tumor_seg.evaluate --config $config --split $split --device $Device
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation failed for $config with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}

Write-Host "All Kaggle 3M model ablations completed successfully." -ForegroundColor Green
