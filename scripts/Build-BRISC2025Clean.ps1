[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [int]$Seed = 42,
    [double]$ValidationFractionOfOfficialTrain = 0.18,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

if ($ValidationFractionOfOfficialTrain -le 0 -or $ValidationFractionOfOfficialTrain -ge 1) {
    throw "ValidationFractionOfOfficialTrain must be between 0 and 1"
}

function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Value))
}

function Get-Sha256Text([string]$Value) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Write-Utf8Json([object]$Value, [string]$Path) {
    $json = $Value | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText(
        $Path,
        $json + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($false))
    )
}

$sourceRoot = Resolve-ProjectPath "DATASET/Segmentation_v2"
$sourceManifest = Resolve-ProjectPath "DATASET/Segmentation_v2_manifest.csv"
$sourceSplitManifest = Resolve-ProjectPath "splits/segmentation_v2_seed42.csv"
$outputRoot = Resolve-ProjectPath "DATASET/BRISC2025_clean"
$cleanManifest = Resolve-ProjectPath "DATASET/BRISC2025_clean_manifest.csv"
$quarantineManifest = Resolve-ProjectPath "DATASET/BRISC2025_clean_quarantine.csv"
$reportPath = Resolve-ProjectPath "DATASET/BRISC2025_clean_report.json"
$splitManifest = Resolve-ProjectPath "splits/brisc2025_clean_seed42.csv"
$splitMetadata = Resolve-ProjectPath "splits/brisc2025_clean_seed42.meta.json"

$outputs = @(
    $outputRoot,
    $cleanManifest,
    $quarantineManifest,
    $reportPath,
    $splitManifest,
    $splitMetadata
)
$existing = @($outputs | Where-Object { Test-Path -LiteralPath $_ })
if ($existing.Count -gt 0 -and -not $Overwrite) {
    throw "BRISC2025 clean outputs already exist; refusing to overwrite: $($existing -join ', ')"
}
if ($Overwrite) {
    foreach ($path in $existing) {
        $resolved = [IO.Path]::GetFullPath($path)
        if (-not $resolved.StartsWith(
                $ProjectRoot + [IO.Path]::DirectorySeparatorChar,
                [StringComparison]::OrdinalIgnoreCase
            )) {
            throw "Refusing to remove output outside project: $resolved"
        }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}

foreach ($required in @($sourceRoot, $sourceManifest, $sourceSplitManifest)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required V2 source artifact does not exist: $required"
    }
}

Write-Output "[1/5] Reading cleaned BRISC provenance from Segmentation_v2"
$v2Rows = @(Import-Csv -LiteralPath $sourceManifest)
$briscRows = @($v2Rows | Where-Object source_dataset -eq "brisc2025")
if ($briscRows.Count -eq 0) {
    throw "No retained BRISC rows found in V2 clean manifest"
}

$groupBySample = @{}
foreach ($row in Import-Csv -LiteralPath $sourceSplitManifest) {
    $groupBySample[$row.sample_id] = $row.group_id
}
foreach ($row in $briscRows) {
    if (-not $groupBySample.ContainsKey($row.sample_id)) {
        throw "Missing similarity group for $($row.sample_id)"
    }
    $row | Add-Member -NotePropertyName GroupId -NotePropertyValue $groupBySample[$row.sample_id]
}

Write-Output "[2/5] Freezing official test and removing train samples linked to test"
$officialTestGroups = @{}
foreach ($row in $briscRows | Where-Object original_split -eq "test") {
    $officialTestGroups[$row.GroupId] = $true
}
$quarantined = @(
    $briscRows |
        Where-Object {
            $_.original_split -eq "train" -and $officialTestGroups.ContainsKey($_.GroupId)
        } |
        ForEach-Object {
            [pscustomobject]@{
                sample_id = $_.sample_id
                source_image_path = $_.source_image_path
                source_mask_path = $_.source_mask_path
                tumor_type = $_.tumor_type
                plane = $_.plane
                similarity_group = $_.GroupId
                reason = "official_train_similar_to_official_test"
            }
        }
)
$quarantineIds = @{}
foreach ($row in $quarantined) { $quarantineIds[$row.sample_id] = $true }
$retained = @($briscRows | Where-Object { -not $quarantineIds.ContainsKey($_.sample_id) })

$assignment = @{}
foreach ($row in $retained | Where-Object original_split -eq "test") {
    $assignment[$row.sample_id] = "test"
}

Write-Output "[3/5] Splitting only official train into grouped train/validation sets"
foreach ($classGroup in $retained | Where-Object original_split -eq "train" | Group-Object tumor_type) {
    $groups = @($classGroup.Group | Group-Object GroupId)
    $targetValidation = [int][Math]::Round(
        $classGroup.Count * $ValidationFractionOfOfficialTrain,
        [MidpointRounding]::AwayFromZero
    )
    $validationCount = 0
    $orderedGroups = @(
        $groups |
            ForEach-Object {
                [pscustomobject]@{
                    Group = $_
                    Size = $_.Count
                    Tie = Get-Sha256Text "$Seed|$($classGroup.Name)|$($_.Name)"
                }
            } |
            Sort-Object Tie
    )
    foreach ($candidate in $orderedGroups) {
        $currentDistance = [Math]::Abs($validationCount - $targetValidation)
        $newDistance = [Math]::Abs(
            ($validationCount + $candidate.Size) - $targetValidation
        )
        $split = if ($newDistance -le $currentDistance) { "val" } else { "train" }
        if ($split -eq "val") { $validationCount += $candidate.Size }
        foreach ($member in $candidate.Group.Group) {
            $assignment[$member.sample_id] = $split
        }
    }
}

if ($assignment.Count -ne $retained.Count) {
    throw "Split assignment count mismatch: $($assignment.Count) vs $($retained.Count)"
}

Write-Output "[4/5] Copying the standalone BRISC-only dataset and writing manifests"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$cleanRows = [Collections.Generic.List[object]]::new()
$splitRows = [Collections.Generic.List[object]]::new()
$copied = 0
foreach ($row in $retained | Sort-Object tumor_type, sample_id) {
    $imageRelative = $row.image_path -replace '/', [IO.Path]::DirectorySeparatorChar
    $maskRelative = $row.mask_path -replace '/', [IO.Path]::DirectorySeparatorChar
    $sourceImage = Join-Path $sourceRoot $imageRelative
    $sourceMask = Join-Path $sourceRoot $maskRelative
    $outputImage = Join-Path $outputRoot $imageRelative
    $outputMask = Join-Path $outputRoot $maskRelative
    New-Item -ItemType Directory -Path (Split-Path -Parent $outputImage) -Force | Out-Null
    Copy-Item -LiteralPath $sourceImage -Destination $outputImage
    Copy-Item -LiteralPath $sourceMask -Destination $outputMask

    $imageHash = (Get-FileHash -LiteralPath $outputImage -Algorithm SHA256).Hash.ToLowerInvariant()
    $maskHash = (Get-FileHash -LiteralPath $outputMask -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($imageHash -ne $row.image_sha256 -or $maskHash -ne $row.mask_sha256) {
        throw "Copied content hash mismatch for $($row.sample_id)"
    }

    $split = $assignment[$row.sample_id]
    $cleanRows.Add([pscustomobject]@{
        sample_id = $row.sample_id
        tumor_type = $row.tumor_type
        plane = $row.plane
        official_split = $row.original_split
        assigned_split = $split
        similarity_group = $row.GroupId
        source_image_path = $row.source_image_path
        source_mask_path = $row.source_mask_path
        image_path = $row.image_path
        mask_path = $row.mask_path
        image_sha256 = $imageHash
        mask_sha256 = $maskHash
        foreground_fraction = $row.foreground_fraction
    })
    $splitRows.Add([pscustomobject]@{
        sample_id = $row.sample_id
        source_id = $row.source_id
        tumor_type = $row.tumor_type
        group_id = $row.GroupId
        split = $split
        image_path = $row.image_path
        mask_path = $row.mask_path
        image_sha256 = $imageHash
        mask_sha256 = $maskHash
    })
    $copied++
    if ($copied % 500 -eq 0) {
        Write-Output "Copied $copied / $($retained.Count) pairs"
    }
}

$cleanRows | Export-Csv -LiteralPath $cleanManifest -NoTypeInformation -Encoding UTF8
$quarantined | Export-Csv -LiteralPath $quarantineManifest -NoTypeInformation -Encoding UTF8
$splitRows | Export-Csv -LiteralPath $splitManifest -NoTypeInformation -Encoding UTF8

Write-Output "[5/5] Validating leakage controls and writing audit metadata"
$imageDuplicates = @($splitRows | Group-Object image_sha256 | Where-Object Count -gt 1)
$sampleDuplicates = @($splitRows | Group-Object sample_id | Where-Object Count -gt 1)
$groupLeaks = @(
    $splitRows |
        Group-Object group_id |
        Where-Object { @($_.Group.split | Sort-Object -Unique).Count -gt 1 }
)
$officialTestViolations = @(
    $cleanRows | Where-Object {
        ($_.official_split -eq "test" -and $_.assigned_split -ne "test") -or
        ($_.official_split -eq "train" -and $_.assigned_split -eq "test")
    }
)
if (
    $imageDuplicates.Count -gt 0 -or
    $sampleDuplicates.Count -gt 0 -or
    $groupLeaks.Count -gt 0 -or
    $officialTestViolations.Count -gt 0
) {
    throw "Leakage validation failed"
}

$counts = @(
    $splitRows |
        Group-Object split |
        Sort-Object Name |
        ForEach-Object { [pscustomobject]@{ split = $_.Name; count = $_.Count } }
)
$classCounts = @(
    $splitRows |
        Group-Object split, tumor_type |
        Sort-Object Name |
        ForEach-Object {
            [pscustomobject]@{
                split = $_.Group[0].split
                tumor_type = $_.Group[0].tumor_type
                count = $_.Count
            }
        }
)
$manifestHash = (Get-FileHash -LiteralPath $splitManifest -Algorithm SHA256).Hash.ToLowerInvariant()
$fingerprintPayload = @(
    $splitRows |
        Sort-Object sample_id |
        ForEach-Object { "$($_.sample_id)|$($_.image_sha256)|$($_.mask_sha256)|$($_.split)" }
) -join "`n"
$datasetFingerprint = Get-Sha256Text $fingerprintPayload
$total = $splitRows.Count
$ratioObject = [ordered]@{}
foreach ($count in $counts) { $ratioObject[$count.split] = $count.count / $total }

$metadata = [ordered]@{
    schema_version = 2
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    seed = $Seed
    ratios = $ratioObject
    split_level = "brisc_official_test_frozen_class_similarity_group"
    num_samples = $total
    counts = $counts
    per_split_class_counts = $classCounts
    dataset_fingerprint = $datasetFingerprint
    manifest_sha256 = $manifestHash
    source_dataset = "BRISC2025 segmentation only"
    official_test_frozen = $true
    official_train_test_similarity_quarantined = $quarantined.Count
    exact_duplicate_groups_remaining = $imageDuplicates.Count
    group_leaks = $groupLeaks.Count
    official_test_assignment_violations = $officialTestViolations.Count
    known_limitation = "BRISC does not expose reliable patient IDs; patient-level independence cannot be proven."
}
Write-Utf8Json $metadata $splitMetadata

$report = [ordered]@{
    schema_version = 1
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    source = [ordered]@{
        dataset = "BRISC2025 segmentation task"
        clean_input = "DATASET/Segmentation_v2_manifest.csv filtered to source_dataset=brisc2025"
        input_pairs = $briscRows.Count
    }
    protocol = [ordered]@{
        official_test_frozen = $true
        validation_source = "official train only"
        validation_fraction_of_official_train = $ValidationFractionOfOfficialTrain
        similarity_groups_indivisible = $true
        official_train_samples_similar_to_test_quarantined = $quarantined.Count
    }
    output = [ordered]@{
        root = "DATASET/BRISC2025_clean"
        retained_pairs = $total
        counts = $counts
        per_split_class_counts = $classCounts
        exact_duplicate_groups_remaining = $imageDuplicates.Count
        similarity_group_leaks = $groupLeaks.Count
        official_test_assignment_violations = $officialTestViolations.Count
    }
    limitations = @(
        "No reliable patient IDs are available.",
        "This build controls exact/perceptual sample leakage and freezes the official test split, but cannot prove patient-level independence.",
        "The subset deliberately inherits the conservative conflict cleaning already applied by Segmentation_v2."
    )
}
Write-Utf8Json $report $reportPath

Write-Output "BRISC2025 clean build complete: $total pairs"
$counts | Format-Table -AutoSize
Write-Output "Quarantined official-train samples linked to test: $($quarantined.Count)"
Write-Output "Manifest SHA-256: $manifestHash"
