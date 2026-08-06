[CmdletBinding()]
param(
    [string]$AuditCsv = "reports/kaggle_3m_audit/samples.csv",
    [string]$SourceManifest = "splits/kaggle_3m_seed42.csv",
    [string]$ManifestPath = "splits/kaggle_3m_multimodal_only_seed42.csv",
    [string]$MetadataPath = "splits/kaggle_3m_multimodal_only_seed42.meta.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-RepoPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}

function Get-BytesSha256Hex([byte[]]$Bytes) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return [BitConverter]::ToString($sha.ComputeHash($Bytes)).Replace("-", "")
    }
    finally {
        $sha.Dispose()
    }
}

function Test-GrayscaleEquivalent($Row) {
    return (
        $Row.RedGreenEqual -eq "True" -and
        $Row.RedBlueEqual -eq "True" -and
        $Row.GreenBlueEqual -eq "True"
    )
}

$auditFull = Resolve-RepoPath $AuditCsv
$sourceManifestFull = Resolve-RepoPath $SourceManifest
$manifestFull = Resolve-RepoPath $ManifestPath
$metadataFull = Resolve-RepoPath $MetadataPath

foreach ($requiredFile in @($auditFull, $sourceManifestFull)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required input file not found: $requiredFile"
    }
}

$auditRows = @(Import-Csv -LiteralPath $auditFull)
$sourceRows = @(Import-Csv -LiteralPath $sourceManifestFull)
if ($auditRows.Count -eq 0 -or $sourceRows.Count -eq 0) {
    throw "Audit CSV and source manifest must both contain rows"
}

$auditBySample = @{}
foreach ($row in $auditRows) {
    $sampleId = "$($row.PatientId)__slice_$($row.SliceNumber)"
    if ($auditBySample.ContainsKey($sampleId)) {
        throw "Duplicate audit sample_id: $sampleId"
    }
    $auditBySample[$sampleId] = $row
}

# Exclude an entire patient if any slice is grayscale-equivalent (R=G=B).
# This guarantees that every retained sample contains three distinct channels.
$excludedPatientGroups = @(
    $auditRows | Group-Object PatientId | Where-Object {
        @($_.Group | Where-Object { Test-GrayscaleEquivalent $_ }).Count -gt 0
    }
)
$excludedPatients = @($excludedPatientGroups | ForEach-Object { $_.Name } | Sort-Object)
$excludedSet = [Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase
)
foreach ($patient in $excludedPatients) {
    [void]$excludedSet.Add($patient)
}

$retainedRows = @($sourceRows | Where-Object { -not $excludedSet.Contains($_.group_id) })
$removedRows = @($sourceRows | Where-Object { $excludedSet.Contains($_.group_id) })
if ($retainedRows.Count -eq 0) {
    throw "Filtering removed every sample"
}

foreach ($row in $retainedRows) {
    if (-not $auditBySample.ContainsKey($row.sample_id)) {
        throw "Manifest sample is missing from audit CSV: $($row.sample_id)"
    }
    if (Test-GrayscaleEquivalent $auditBySample[$row.sample_id]) {
        throw "Retained sample is still grayscale-equivalent: $($row.sample_id)"
    }
}

$splitNames = @("train", "val", "test")
foreach ($split in $splitNames) {
    if (@($retainedRows | Where-Object { $_.split -eq $split }).Count -eq 0) {
        throw "Filtered split has no samples: $split"
    }
}

[IO.Directory]::CreateDirectory((Split-Path -Parent $manifestFull)) | Out-Null
[IO.Directory]::CreateDirectory((Split-Path -Parent $metadataFull)) | Out-Null
$retainedRows | Export-Csv -LiteralPath $manifestFull -NoTypeInformation -Encoding utf8
$manifestHash = (Get-FileHash -LiteralPath $manifestFull -Algorithm SHA256).Hash.ToLowerInvariant()

$fingerprintBuilder = [Text.StringBuilder]::new()
foreach ($row in ($retainedRows | Sort-Object sample_id)) {
    [void]$fingerprintBuilder.Append(
        "$($row.sample_id)`0$($row.image_path)`0$($row.mask_path)`0$($row.image_sha256)`0$($row.mask_sha256)`n"
    )
}
$fingerprint = (
    Get-BytesSha256Hex ([Text.Encoding]::UTF8.GetBytes($fingerprintBuilder.ToString()))
).ToLowerInvariant()

$counts = [ordered]@{}
$positiveCounts = [ordered]@{}
$patientCounts = [ordered]@{}
$perSplit = [ordered]@{}
foreach ($split in $splitNames) {
    $splitRows = @($retainedRows | Where-Object { $_.split -eq $split })
    $positive = @(
        $splitRows | Where-Object {
            [int64]$auditBySample[$_.sample_id].ForegroundPixels -gt 0
        }
    ).Count
    $patients = @($splitRows.group_id | Sort-Object -Unique).Count
    $counts[$split] = $splitRows.Count
    $positiveCounts[$split] = $positive
    $patientCounts[$split] = $patients
    $perSplit[$split] = [ordered]@{
        samples = $splitRows.Count
        patients = $patients
        positive_masks = $positive
        empty_masks = $splitRows.Count - $positive
    }
}

$excludedDetails = @(
    foreach ($patient in $excludedPatients) {
        $patientRows = @($sourceRows | Where-Object { $_.group_id -eq $patient })
        $positive = @(
            $patientRows | Where-Object {
                [int64]$auditBySample[$_.sample_id].ForegroundPixels -gt 0
            }
        ).Count
        [ordered]@{
            patient_id = $patient
            split = ($patientRows.split | Select-Object -First 1)
            samples = $patientRows.Count
            positive_masks = $positive
            empty_masks = $patientRows.Count - $positive
        }
    }
)

$metadata = [ordered]@{
    schema_version = 1
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    seed = 42
    ratios = [ordered]@{ train = 0.70; val = 0.15; test = 0.15 }
    split_level = "patient"
    split_assignment = "preserved_from_source_manifest"
    source_manifest = $SourceManifest
    source_manifest_sha256 = (
        Get-FileHash -LiteralPath $sourceManifestFull -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    cleaning_rule = "exclude entire patient when any audited slice has R=G=B"
    num_samples = $retainedRows.Count
    num_patients = @($retainedRows.group_id | Sort-Object -Unique).Count
    num_excluded_samples = $removedRows.Count
    num_excluded_patients = $excludedPatients.Count
    excluded_patients = $excludedDetails
    counts = $counts
    positive_mask_counts = $positiveCounts
    patient_counts = $patientCounts
    per_split = $perSplit
    dataset_fingerprint = $fingerprint
    manifest_sha256 = $manifestHash
    mask_binarization = "mask >= 128"
    channel_mode = "rgb_multimodal_only"
    source_audit_csv = $AuditCsv
}
$metadataJson = ($metadata | ConvertTo-Json -Depth 10) + "`n"
[IO.File]::WriteAllText($metadataFull, $metadataJson, [Text.UTF8Encoding]::new($false))

Write-Host "Created multimodal-only manifest: $manifestFull" -ForegroundColor Green
Write-Host "Created metadata: $metadataFull" -ForegroundColor Green
Write-Host "Excluded patients=$($excludedPatients.Count), samples=$($removedRows.Count)" -ForegroundColor Yellow
foreach ($detail in $excludedDetails) {
    Write-Host "  $($detail.patient_id): split=$($detail.split), samples=$($detail.samples)"
}
foreach ($split in $splitNames) {
    Write-Host (
        "$split samples=$($counts[$split]) patients=$($patientCounts[$split]) " +
        "positive=$($positiveCounts[$split]) empty=$($counts[$split] - $positiveCounts[$split])"
    )
}

