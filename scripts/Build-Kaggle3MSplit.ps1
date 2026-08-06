[CmdletBinding()]
param(
    [string]$AuditCsv = "reports/kaggle_3m_audit/samples.csv",
    [string]$DatasetRoot = "DATASET/kaggle_3m",
    [string]$ManifestPath = "splits/kaggle_3m_seed42.csv",
    [string]$MetadataPath = "splits/kaggle_3m_seed42.meta.json",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
function Resolve-RepoPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}
function Get-BytesSha256Hex([byte[]]$Bytes) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash($Bytes)).Replace("-", "") }
    finally { $sha.Dispose() }
}

$auditFull = Resolve-RepoPath $AuditCsv
$datasetFull = Resolve-RepoPath $DatasetRoot
$manifestFull = Resolve-RepoPath $ManifestPath
$metadataFull = Resolve-RepoPath $MetadataPath
if (-not (Test-Path -LiteralPath $auditFull -PathType Leaf)) {
    throw "Audit sample table not found: $auditFull. Run scripts/Audit-Kaggle3M.ps1 first."
}
if (-not (Test-Path -LiteralPath $datasetFull -PathType Container)) {
    throw "Dataset root not found: $datasetFull"
}

$rows = @(Import-Csv -LiteralPath $auditFull)
if ($rows.Count -eq 0) { throw "Audit sample table is empty: $auditFull" }
$ratios = [ordered]@{ train = 0.70; val = 0.15; test = 0.15 }
$splitNames = @("train", "val", "test")
$totalSamples = $rows.Count
$totalPositive = @($rows | Where-Object { [int64]$_.ForegroundPixels -gt 0 }).Count

$patientGroups = @(
    $rows | Group-Object PatientId | ForEach-Object {
        $members = @($_.Group)
        [pscustomobject]@{
            PatientId = $_.Name
            Count = $members.Count
            Positive = @($members | Where-Object { [int64]$_.ForegroundPixels -gt 0 }).Count
            Tie = Get-BytesSha256Hex ([Text.Encoding]::UTF8.GetBytes("$Seed|$($_.Name)"))
        }
    } | Sort-Object @{ Expression = { $_.Count }; Descending = $true },
        @{ Expression = { $_.Positive }; Descending = $true }, Tie
)
if ($patientGroups.Count -lt 3) { throw "At least three patient folders are required" }

$sampleCounts = @{ train = 0; val = 0; test = 0 }
$positiveCounts = @{ train = 0; val = 0; test = 0 }
$patientCounts = @{ train = 0; val = 0; test = 0 }
$assignment = @{}
foreach ($group in $patientGroups) {
    $bestSplit = $null
    $bestScore = [double]::PositiveInfinity
    foreach ($candidate in $splitNames) {
        $score = 0.0
        foreach ($split in $splitNames) {
            $candidateSamples = $sampleCounts[$split]
            $candidatePositive = $positiveCounts[$split]
            if ($split -eq $candidate) {
                $candidateSamples += $group.Count
                $candidatePositive += $group.Positive
            }
            $sampleTarget = $ratios[$split] * $totalSamples
            $positiveTarget = $ratios[$split] * $totalPositive
            $score += [Math]::Pow(($candidateSamples - $sampleTarget) / $sampleTarget, 2)
            $score += [Math]::Pow(($candidatePositive - $positiveTarget) / $positiveTarget, 2)
        }
        if ($score -lt $bestScore - 1e-12) {
            $bestScore = $score
            $bestSplit = $candidate
        }
    }
    $assignment[$group.PatientId] = $bestSplit
    $sampleCounts[$bestSplit] += $group.Count
    $positiveCounts[$bestSplit] += $group.Positive
    $patientCounts[$bestSplit] += 1
}
foreach ($split in $splitNames) {
    if ($patientCounts[$split] -eq 0) { throw "Split $split has no patients" }
}

$datasetPrefix = ($DatasetRoot.TrimEnd('/', '\') -replace '\\', '/') + '/'
$sourceId = 0
$manifestRows = foreach ($row in ($rows | Sort-Object PatientId, @{ Expression = { [int]$_.SliceNumber } })) {
    $sourceId += 1
    $imageRelative = ($row.ImageRelative -replace '\\', '/')
    $maskRelative = ($row.MaskRelative -replace '\\', '/')
    if ($imageRelative.StartsWith($datasetPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        $imageRelative = $imageRelative.Substring($datasetPrefix.Length)
    }
    if ($maskRelative.StartsWith($datasetPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        $maskRelative = $maskRelative.Substring($datasetPrefix.Length)
    }
    $imageFull = Join-Path $datasetFull ($imageRelative -replace '/', [IO.Path]::DirectorySeparatorChar)
    $maskFull = Join-Path $datasetFull ($maskRelative -replace '/', [IO.Path]::DirectorySeparatorChar)
    if (-not (Test-Path -LiteralPath $imageFull -PathType Leaf)) { throw "Missing image: $imageFull" }
    if (-not (Test-Path -LiteralPath $maskFull -PathType Leaf)) { throw "Missing mask: $maskFull" }
    [pscustomobject][ordered]@{
        sample_id = "$($row.PatientId)__slice_$($row.SliceNumber)"
        source_id = $sourceId
        tumor_type = "LGG"
        group_id = $row.PatientId
        split = $assignment[$row.PatientId]
        image_path = $imageRelative
        mask_path = $maskRelative
        image_sha256 = $row.ImageFileSha256
        mask_sha256 = $row.MaskFileSha256
    }
}

$manifestDirectory = Split-Path -Parent $manifestFull
$metadataDirectory = Split-Path -Parent $metadataFull
[IO.Directory]::CreateDirectory($manifestDirectory) | Out-Null
[IO.Directory]::CreateDirectory($metadataDirectory) | Out-Null
$manifestRows | Export-Csv -LiteralPath $manifestFull -NoTypeInformation -Encoding utf8
$manifestHash = (Get-FileHash -LiteralPath $manifestFull -Algorithm SHA256).Hash.ToLowerInvariant()

$fingerprintBuilder = [Text.StringBuilder]::new()
foreach ($row in ($manifestRows | Sort-Object sample_id)) {
    [void]$fingerprintBuilder.Append(
        "$($row.sample_id)`0$($row.image_path)`0$($row.mask_path)`0$($row.image_sha256)`0$($row.mask_sha256)`n"
    )
}
$fingerprintBytes = [Text.Encoding]::UTF8.GetBytes($fingerprintBuilder.ToString())
$fingerprint = (Get-BytesSha256Hex $fingerprintBytes).ToLowerInvariant()

$splitStats = [ordered]@{}
foreach ($split in $splitNames) {
    $splitStats[$split] = [ordered]@{
        samples = $sampleCounts[$split]
        patients = $patientCounts[$split]
        positive_masks = $positiveCounts[$split]
        empty_masks = $sampleCounts[$split] - $positiveCounts[$split]
    }
}
$metadata = [ordered]@{
    schema_version = 1
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    seed = $Seed
    ratios = $ratios
    split_level = "patient"
    group_csv = $null
    num_samples = $totalSamples
    num_patients = $patientGroups.Count
    counts = $sampleCounts
    positive_mask_counts = $positiveCounts
    patient_counts = $patientCounts
    per_split = $splitStats
    dataset_fingerprint = $fingerprint
    manifest_sha256 = $manifestHash
    mask_binarization = "mask >= 128"
    channel_mode = "flair_green"
    source_audit_csv = $AuditCsv
}
$metadataJson = ($metadata | ConvertTo-Json -Depth 8) + "`n"
[IO.File]::WriteAllText($metadataFull, $metadataJson, [Text.UTF8Encoding]::new($false))

Write-Host "Created $manifestFull"
Write-Host "Created $metadataFull"
Write-Host "manifest_sha256=$manifestHash"
foreach ($split in $splitNames) {
    Write-Host "$split samples=$($sampleCounts[$split]) patients=$($patientCounts[$split]) positive=$($positiveCounts[$split]) empty=$($sampleCounts[$split] - $positiveCounts[$split])"
}
