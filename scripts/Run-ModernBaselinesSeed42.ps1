[CmdletBinding()]
param(
    [string]$Python = "python",
    [ValidateSet("Both", "NnUNet", "TransUNet")]
    [string]$Baseline = "Both",
    [ValidateSet("All", "Prepare", "SmokeTest", "Train", "Test", "Report")]
    [string]$Stage = "All",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$transConfig = "configs/kaggle_3m_multimodal_only_transunet_2d_basic_no_augmentation_seed42.yaml"
$nnConfig = "configs/nnunetv2_2d_kaggle_3m_clean_no_augmentation_seed42.yaml"
$nnRun = Join-Path $projectRoot "runs/nnunetv2_2d_kaggle_3m_clean_no_augmentation_seed42"
$nnWorkspace = Join-Path $nnRun "workspace"
$nnRaw = Join-Path $nnWorkspace "nnUNet_raw"
$nnPreprocessed = Join-Path $nnWorkspace "nnUNet_preprocessed"
$nnResults = Join-Path $nnWorkspace "nnUNet_results"
$datasetFolder = "Dataset501_Kaggle3MClean"

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Confirm-Python {
    try {
        $version = & $Python --version 2>&1
    }
    catch {
        throw (
            "Python executable is unavailable: $Python. Activate the project environment " +
            "or pass -Python <path-to-python.exe>."
        )
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python executable failed: $Python"
    }
    Write-Host $version -ForegroundColor DarkGray
}

function Resolve-PythonTool {
    param([Parameter(Mandatory=$true)][string]$Name)
    $pythonExecutable = (& $Python -c "import sys; print(sys.executable)").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $pythonExecutable) {
        throw "Could not resolve the active Python executable"
    }
    $pythonBin = Split-Path -Parent $pythonExecutable
    # Windows virtualenv/Conda executables live in Scripts, whereas POSIX
    # environments place them beside the Python binary. Check both layouts.
    $candidates = @(
        (Join-Path $pythonBin "${Name}.exe"),
        (Join-Path (Join-Path $pythonBin "Scripts") "${Name}.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    throw (
        "Required nnU-Net command is unavailable: $Name. Install the optional baseline " +
        "dependencies with: $Python -m pip install -r requirements-nnunet.txt"
    )
}

function Invoke-CheckedTool {
    param(
        [Parameter(Mandatory=$true)][string]$Tool,
        [Parameter(Mandatory=$true)][string[]]$Arguments
    )
    & $Tool @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Tool $($Arguments -join ' ')"
    }
}

function Set-NnUNetEnvironment {
    $env:nnUNet_raw = $nnRaw
    $env:nnUNet_preprocessed = $nnPreprocessed
    $env:nnUNet_results = $nnResults
    $env:PYTHONHASHSEED = "42"
    # Windows-safe resource limits. nnUNet_n_proc_DA=0 selects the official
    # single-threaded augmenter and avoids WinError 1455 shared-memory maps.
    $env:nnUNet_n_proc_DA = "0"
    $env:nnUNet_def_n_proc = "1"
    $env:nnUNet_npp = "1"
    $env:nnUNet_nps = "1"
}

function Invoke-TransUNet {
    $runDir = Join-Path $projectRoot (
        "runs/kaggle_3m_multimodal_only_transunet_2d_basic_no_augmentation_seed42"
    )
    $bestCheckpoint = Join-Path $runDir "checkpoints/best.pt"
    $lastCheckpoint = Join-Path $runDir "checkpoints/last.pt"
    $summaryPath = Join-Path $runDir "training_summary.json"
    $metricsPath = Join-Path $runDir "metrics.jsonl"
    $testMetrics = Join-Path $runDir "test_metrics.json"

    if ($Stage -in @("All", "Prepare")) {
        Write-Host "Rebuilding and verifying the fixed clean patient split..." -ForegroundColor Cyan
        & (Join-Path $PSScriptRoot "Build-Kaggle3MMultimodalOnlySplit.ps1")
        if ($LASTEXITCODE -ne 0) {
            throw "Clean patient split verification failed"
        }
    }
    if ($Stage -in @("All", "Train")) {
        if ((Test-Path -LiteralPath $summaryPath -PathType Leaf) -and
            (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
            Write-Host "Skipping completed TransUNet training" -ForegroundColor DarkYellow
        }
        elseif (Test-Path -LiteralPath $metricsPath -PathType Leaf) {
            if (-not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
                throw "Incomplete TransUNet run has no last checkpoint: $runDir"
            }
            Write-Host "Resuming TransUNet seed 42..." -ForegroundColor Yellow
            Invoke-CheckedPython -Arguments @(
                "-m", "brain_tumor_seg.train", "--config", $transConfig,
                "--device", $Device, "--resume", $lastCheckpoint
            )
        }
        else {
            Write-Host "Training basic TransUNet seed 42..." -ForegroundColor Cyan
            Invoke-CheckedPython -Arguments @(
                "-m", "brain_tumor_seg.train", "--config", $transConfig,
                "--device", $Device
            )
        }
    }
    if ($Stage -in @("All", "Test")) {
        if (Test-Path -LiteralPath $testMetrics -PathType Leaf) {
            Write-Host "Skipping completed TransUNet test" -ForegroundColor DarkYellow
        }
        else {
            if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
                throw "TransUNet best checkpoint not found: $bestCheckpoint"
            }
            Write-Host "Testing basic TransUNet on the fixed test patients..." `
                -ForegroundColor Green
            Invoke-CheckedPython -Arguments @(
                "-m", "brain_tumor_seg.evaluate", "--config", $transConfig,
                "--split", "test", "--device", $Device
            )
        }
    }
}

function Invoke-NnUNet {
    Set-NnUNetEnvironment
    $rawDataset = Join-Path $nnRaw $datasetFolder
    $preprocessedDataset = Join-Path $nnPreprocessed $datasetFolder
    $plansPath = Join-Path $preprocessedDataset "nnUNetPlans.json"
    $sourceSplit = Join-Path $rawDataset "splits_final.json"
    $fixedSplit = Join-Path $preprocessedDataset "splits_final.json"
    $predictionDir = Join-Path $nnRun "predictions/test"
    $testMetrics = Join-Path $nnRun "test_metrics.json"
    $trainerOutput = Join-Path $nnResults (
        "$datasetFolder/nnUNetTrainerNoDA__nnUNetPlans__2d/fold_0"
    )
    $finalCheckpointPath = Join-Path $trainerOutput "checkpoint_final.pth"
    $latestCheckpointPath = Join-Path $trainerOutput "checkpoint_latest.pth"
    $bestCheckpointPath = Join-Path $trainerOutput "checkpoint_best.pth"
    $commonTrainArguments = @(
        "scripts/train_nnunet_seeded.py",
        "--dataset-id", "501", "--configuration", "2d", "--fold", "0",
        "--trainer", "nnUNetTrainerNoDA", "--plans", "nnUNetPlans",
        "--seed", "42", "--device", $Device,
        "--batch-size", "4", "--num-data-workers", "0",
        "--num-export-workers", "1"
    )

    if ($Stage -in @("All", "Prepare")) {
        Write-Host "Preparing the official nnU-Net v2 raw dataset..." -ForegroundColor Cyan
        Invoke-CheckedPython -Arguments @(
            "scripts/prepare_nnunet_kaggle3m.py", "--config", $nnConfig
        )
        if (-not (Test-Path -LiteralPath $plansPath -PathType Leaf)) {
            $planTool = Resolve-PythonTool -Name "nnUNetv2_plan_and_preprocess"
            Write-Host "Planning and preprocessing official nnU-Net 2D..." `
                -ForegroundColor Cyan
            Invoke-CheckedTool -Tool $planTool -Arguments @(
                "-d", "501", "-c", "2d", "--verify_dataset_integrity"
            )
        }
        if (-not (Test-Path -LiteralPath $sourceSplit -PathType Leaf)) {
            throw "Fixed source split is missing: $sourceSplit"
        }
        Copy-Item -LiteralPath $sourceSplit -Destination $fixedSplit -Force
        Write-Host "Installed the fixed 85/9-patient train/validation fold." `
            -ForegroundColor Green
    }

    if ($Stage -in @("All", "SmokeTest") -and
        -not (Test-Path -LiteralPath $finalCheckpointPath -PathType Leaf)) {
        if (-not (Test-Path -LiteralPath $fixedSplit -PathType Leaf)) {
            throw "nnU-Net preprocessing/fixed split is missing. Run -Stage Prepare first."
        }
        Write-Host "Running one isolated nnU-Net forward/backward batch..." `
            -ForegroundColor Cyan
        Invoke-CheckedPython -Arguments ($commonTrainArguments + "--smoke-test")
    }

    if ($Stage -in @("All", "Train")) {
        if (-not (Test-Path -LiteralPath $fixedSplit -PathType Leaf)) {
            throw "nnU-Net preprocessing/fixed split is missing. Run -Stage Prepare first."
        }
        if (Test-Path -LiteralPath $finalCheckpointPath -PathType Leaf) {
            Write-Host "Skipping completed nnU-Net training" -ForegroundColor DarkYellow
        }
        else {
            $arguments = $commonTrainArguments
            if (Test-Path -LiteralPath $latestCheckpointPath -PathType Leaf) {
                $arguments += "--continue-training"
                Write-Host "Resuming official nnU-Net 2D seed 42..." `
                    -ForegroundColor Yellow
            }
            else {
                Write-Host "Training official nnU-Net 2D seed 42 without augmentation..." `
                    -ForegroundColor Cyan
            }
            Invoke-CheckedPython -Arguments $arguments
        }
    }

    if ($Stage -in @("All", "Test")) {
        if (-not (Test-Path -LiteralPath $bestCheckpointPath -PathType Leaf)) {
            throw "nnU-Net best validation checkpoint not found: $bestCheckpointPath"
        }
        $predictionCount = @(
            Get-ChildItem -LiteralPath $predictionDir -Filter "*.png" -File `
                -ErrorAction SilentlyContinue
        ).Count
        if ($predictionCount -ne 525) {
            $predictTool = Resolve-PythonTool -Name "nnUNetv2_predict"
            if (-not (Test-Path -LiteralPath $predictionDir)) {
                New-Item -ItemType Directory -Path $predictionDir | Out-Null
            }
            Write-Host "Predicting the fixed 525-slice test set with nnU-Net..." `
                -ForegroundColor Green
            Invoke-CheckedTool -Tool $predictTool -Arguments @(
                "-i", (Join-Path $rawDataset "imagesTs"),
                "-o", $predictionDir,
                "-d", "501", "-c", "2d", "-f", "0",
                "-tr", "nnUNetTrainerNoDA", "-p", "nnUNetPlans",
                "-chk", "checkpoint_best.pth", "-device", $Device,
                "-npp", "1", "-nps", "1",
                "--disable_tta"
            )
        }
        else {
            Write-Host "Skipping completed nnU-Net prediction" -ForegroundColor DarkYellow
        }
        if (Test-Path -LiteralPath $testMetrics -PathType Leaf) {
            Write-Host "Skipping completed nnU-Net metric aggregation" `
                -ForegroundColor DarkYellow
        }
        else {
            Invoke-CheckedPython -Arguments @(
                "scripts/evaluate_nnunet_predictions.py",
                "--config", $nnConfig,
                "--predictions", $predictionDir
            )
        }
    }
}

function Invoke-Report {
    $reportPath = Join-Path $projectRoot "reports/kaggle_3m_modern_baselines_seed42.md"
    Write-Host "Generating the seed-42 baseline report..." -ForegroundColor Cyan
    Invoke-CheckedPython -Arguments @(
        "scripts/generate_modern_baselines_report.py",
        "--output", $reportPath
    )
    Write-Host "Report written to: $reportPath" -ForegroundColor Green
}

Push-Location $projectRoot
try {
    Confirm-Python
    $sourcePath = Join-Path $projectRoot "src"
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$sourcePath$([IO.Path]::PathSeparator)$env:PYTHONPATH"
    }
    else {
        $env:PYTHONPATH = $sourcePath
    }
    $env:PYTHONHASHSEED = "42"

    if ($Baseline -in @("Both", "NnUNet")) {
        Invoke-NnUNet
    }
    if ($Baseline -in @("Both", "TransUNet")) {
        Invoke-TransUNet
    }
    if ($Stage -in @("All", "Report")) {
        Invoke-Report
    }
}
finally {
    Pop-Location
}

Write-Host "Requested seed-42 modern baseline stages completed." -ForegroundColor Green
