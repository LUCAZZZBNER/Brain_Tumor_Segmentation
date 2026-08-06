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

$experiments = @(
    [pscustomobject]@{
        Name = "M4-P-minus-A"
        BaseConfig = (
            "configs/kaggle_3m_multimodal_only_m4_p_minus_a_flair_resnet34_unet.yaml"
        )
        RunStem = "kaggle_3m_multimodal_only_m4_p_minus_a_flair_resnet34_unet"
    }
    [pscustomobject]@{
        Name = "M4-P-minus-B"
        BaseConfig = (
            "configs/" +
            "kaggle_3m_multimodal_only_m4_p_minus_b_rgb_resnet34_unet_no_augmentation.yaml"
        )
        RunStem = (
            "kaggle_3m_multimodal_only_m4_p_minus_b_rgb_resnet34_unet_no_augmentation"
        )
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
        throw (
            "Python executable is unavailable: $Python. " +
            "Activate the project environment or pass -Python <path>."
        )
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python executable failed with exit code $LASTEXITCODE`: $Python"
    }
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

function Get-ConfigPath($Experiment, [int]$Seed) {
    if ($Seed -eq 42) {
        return $Experiment.BaseConfig
    }
    $baseLeaf = [IO.Path]::GetFileNameWithoutExtension($Experiment.BaseConfig)
    return "configs/${baseLeaf}_seed${Seed}.yaml"
}

function Write-SeedConfig($Experiment, [int]$Seed) {
    if ($Seed -eq 42) {
        return
    }
    $basePath = Join-Path $projectRoot $Experiment.BaseConfig
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
    $configRelative = Get-ConfigPath $Experiment $Seed
    $configPath = Join-Path $projectRoot $configRelative
    [IO.File]::WriteAllText($configPath, $derived, [Text.UTF8Encoding]::new($false))
    Write-Host "Generated $configRelative" -ForegroundColor DarkCyan
}

function Confirm-CleanSplit {
    Write-Host "Rebuilding deterministic multimodal-only split..." -ForegroundColor Cyan
    & $buildScript
}

Push-Location $projectRoot
try {
    Confirm-Seeds
    $seedLabel = $Seeds -join "_"
    $reportPath = (
        "reports/" +
        "kaggle_3m_multimodal_only_m4_p_final_component_ablation_seeds${seedLabel}.md"
    )

    Confirm-CleanSplit
    foreach ($seed in $Seeds) {
        foreach ($experiment in $experiments) {
            Write-SeedConfig $experiment $seed
        }
    }

    if ($Stage -ne "Configs") {
        Confirm-Python
    }

    if ($Stage -in @("All", "Train")) {
        foreach ($seed in $Seeds) {
            foreach ($experiment in $experiments) {
                $config = Get-ConfigPath $experiment $seed
                $runDir = Join-Path $projectRoot "runs\$($experiment.RunStem)_seed$seed"
                $summaryPath = Join-Path $runDir "training_summary.json"
                $bestCheckpoint = Join-Path $runDir "checkpoints\best.pt"
                $lastCheckpoint = Join-Path $runDir "checkpoints\last.pt"
                $metricsPath = Join-Path $runDir "metrics.jsonl"

                if ((Test-Path -LiteralPath $summaryPath -PathType Leaf) -and
                    (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
                    Write-Host (
                        "Skipping completed training: $($experiment.Name) seed=$seed"
                    ) -ForegroundColor DarkYellow
                    continue
                }
                if (Test-Path -LiteralPath $metricsPath -PathType Leaf) {
                    if (-not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
                        throw "Incomplete run has no last checkpoint: $runDir"
                    }
                    Write-Host (
                        "Resuming $($experiment.Name) seed=$seed"
                    ) -ForegroundColor Yellow
                    Invoke-CheckedPython -Arguments @(
                        "-m", "brain_tumor_seg.train",
                        "--config", $config,
                        "--device", $Device,
                        "--resume", $lastCheckpoint
                    )
                    continue
                }

                Write-Host (
                    "Training $($experiment.Name) seed=$seed"
                ) -ForegroundColor Cyan
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
            foreach ($experiment in $experiments) {
                $config = Get-ConfigPath $experiment $seed
                $runDir = Join-Path $projectRoot "runs\$($experiment.RunStem)_seed$seed"
                $bestCheckpoint = Join-Path $runDir "checkpoints\best.pt"
                $testMetrics = Join-Path $runDir "test_metrics.json"
                $sampleMetrics = Join-Path $runDir "evaluation\test\samples.csv"
                if ((Test-Path -LiteralPath $testMetrics -PathType Leaf) -and
                    (Test-Path -LiteralPath $sampleMetrics -PathType Leaf)) {
                    Write-Host (
                        "Skipping completed test: $($experiment.Name) seed=$seed"
                    ) -ForegroundColor DarkYellow
                    continue
                }
                if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
                    throw "Best checkpoint not found: $bestCheckpoint"
                }

                Write-Host (
                    "Testing $($experiment.Name) seed=$seed"
                ) -ForegroundColor Green
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
        $curvePngPath = (
            "reports/figures/" +
            "kaggle_3m_multimodal_only_m4_p_training_curves_seeds${seedLabel}.png"
        )
        $curvePdfPath = (
            "reports/figures/" +
            "kaggle_3m_multimodal_only_m4_p_training_curves_seeds${seedLabel}.pdf"
        )
        $plotArguments = @(
            "scripts/plot_kaggle3m_clean_m4_p_training_curves.py",
            "--output", $curvePngPath,
            "--pdf-output", $curvePdfPath,
            "--seeds"
        )
        $plotArguments += @($Seeds | ForEach-Object { [string]$_ })
        Write-Host "Generating 2x2 training-curve figure..." -ForegroundColor Cyan
        Invoke-CheckedPython -Arguments $plotArguments

        $reportArguments = @(
            "scripts/summarize_kaggle3m_clean_m4_p_final_component_ablation.py",
            "--output", $reportPath,
            "--seeds"
        )
        $reportArguments += @($Seeds | ForEach-Object { [string]$_ })
        Write-Host "Generating M4-P final-component ablation report..." `
            -ForegroundColor Cyan
        Invoke-CheckedPython -Arguments $reportArguments
    }
}
finally {
    Pop-Location
}

if ($Stage -eq "All") {
    Write-Host "Completed M4-P final-component ablation." -ForegroundColor Green
    Write-Host "Report: $reportPath" -ForegroundColor Green
}
