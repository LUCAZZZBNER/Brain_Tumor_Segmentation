param(
    [ValidateSet("Train", "Test", "All")]
    [string]$Stage = "Train",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$configs = @(
    "configs/v2_unet_no_augmentation.yaml",
    "configs/v2_attention_unet_no_augmentation.yaml",
    "configs/v2_aspp_unet_no_augmentation.yaml",
    "configs/v2_resnet34_unet_no_augmentation.yaml"
)

function Invoke-CheckedPython {
    param([string[]]$Arguments)

    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Push-Location $projectRoot
try {
    if ($Stage -in @("Train", "All")) {
        foreach ($config in $configs) {
            Write-Host "Training: $config"
            Invoke-CheckedPython -Arguments @(
                "-m", "brain_tumor_seg.train",
                "--config", $config,
                "--device", $Device
            )
        }
    }

    if ($Stage -in @("Test", "All")) {
        foreach ($config in $configs) {
            Write-Host "Testing best checkpoint: $config"
            Invoke-CheckedPython -Arguments @(
                "-m", "brain_tumor_seg.evaluate",
                "--config", $config,
                "--split", "test",
                "--device", $Device
            )
        }
    }
}
finally {
    Pop-Location
}
