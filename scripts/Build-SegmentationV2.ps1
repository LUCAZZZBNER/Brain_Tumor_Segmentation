[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$OutputRoot = "DATASET/Segmentation_v2",
    [string]$CleanManifest = "DATASET/Segmentation_v2_manifest.csv",
    [string]$QuarantineManifest = "DATASET/Segmentation_v2_quarantine.csv",
    [string]$ReportPath = "DATASET/Segmentation_v2_report.json",
    [string]$SplitManifest = "splits/segmentation_v2_seed42.csv",
    [string]$SplitMetadata = "splits/segmentation_v2_seed42.meta.json",
    [int]$TargetSize = 512,
    [int]$Seed = 42,
    [double]$DuplicateMaskIoU = 0.95,
    [int]$CrossDatasetHammingDistance = 12,
    [double]$CrossDatasetCorrelation = 0.95,
    [int]$SplitGroupHammingDistance = 8,
    [double]$SplitGroupCorrelation = 0.98,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ProjectRoot = Split-Path -Parent $scriptDirectory
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

function Resolve-ProjectPath([string]$PathValue) {
    if ([IO.Path]::IsPathRooted($PathValue)) {
        return [IO.Path]::GetFullPath($PathValue)
    }
    return [IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
}

function To-ProjectRelativePath([string]$AbsolutePath) {
    $rootUri = [Uri](($ProjectRoot.TrimEnd('\') + '\'))
    $pathUri = [Uri]([IO.Path]::GetFullPath($AbsolutePath))
    return [Uri]::UnescapeDataString($rootUri.MakeRelativeUri($pathUri).ToString())
}

function Write-Utf8Json([object]$Value, [string]$PathValue) {
    $json = $Value | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText($PathValue, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
}

function Get-ClassSourceCounts([object[]]$Rows) {
    return @(
        $Rows |
            Group-Object TumorType, SourceDataset |
            Sort-Object Name |
            ForEach-Object {
                [pscustomobject]@{
                    tumor_type = $_.Group[0].TumorType
                    source_dataset = $_.Group[0].SourceDataset
                    count = $_.Count
                }
            }
    )
}

$outputRootAbsolute = Resolve-ProjectPath $OutputRoot
$cleanManifestAbsolute = Resolve-ProjectPath $CleanManifest
$quarantineManifestAbsolute = Resolve-ProjectPath $QuarantineManifest
$reportAbsolute = Resolve-ProjectPath $ReportPath
$splitManifestAbsolute = Resolve-ProjectPath $SplitManifest
$splitMetadataAbsolute = Resolve-ProjectPath $SplitMetadata
$allOutputs = @(
    $outputRootAbsolute,
    $cleanManifestAbsolute,
    $quarantineManifestAbsolute,
    $reportAbsolute,
    $splitManifestAbsolute,
    $splitMetadataAbsolute
)

$existingOutputs = @($allOutputs | Where-Object { Test-Path -LiteralPath $_ })
if ($existingOutputs.Count -gt 0 -and -not $Overwrite) {
    throw "V2 outputs already exist. Refusing to overwrite: $($existingOutputs -join ', ')"
}
if ($Overwrite) {
    foreach ($path in $existingOutputs) {
        $resolved = [IO.Path]::GetFullPath($path)
        if (-not $resolved.StartsWith($ProjectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove an output outside the project: $resolved"
        }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}

New-Item -ItemType Directory -Path $outputRootAbsolute -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $cleanManifestAbsolute) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $splitManifestAbsolute) -Force | Out-Null

Add-Type -AssemblyName System.Drawing
Add-Type -Path (Join-Path $PSScriptRoot "V2ImageTools.cs") -ReferencedAssemblies System.Drawing

Write-Output "[1/7] Reading source manifests"
$candidates = [Collections.Generic.List[object]]::new()
$oldCandidates = [Collections.Generic.List[object]]::new()
$briscCandidates = [Collections.Generic.List[object]]::new()

$oldRoot = Join-Path $ProjectRoot "DATASET\Segmentation"
$oldManifestPath = Join-Path $ProjectRoot "splits\baseline_seed42.csv"
foreach ($row in Import-Csv -LiteralPath $oldManifestPath) {
    $imageAbsolute = Join-Path $oldRoot ($row.image_path -replace '/', '\')
    $maskAbsolute = Join-Path $oldRoot ($row.mask_path -replace '/', '\')
    if (-not (Test-Path -LiteralPath $imageAbsolute -PathType Leaf) -or
        -not (Test-Path -LiteralPath $maskAbsolute -PathType Leaf)) {
        throw "Missing original pair: $imageAbsolute, $maskAbsolute"
    }
    $candidate = [pscustomobject]@{
        Index = $candidates.Count
        CandidateId = "original:$($row.sample_id)"
        SourceDataset = "original"
        TumorType = $row.tumor_type
        ImageAbsolute = $imageAbsolute
        MaskAbsolute = $maskAbsolute
        SourceImagePath = "DATASET/Segmentation/$($row.image_path)"
        SourceMaskPath = "DATASET/Segmentation/$($row.mask_path)"
        ImageSha256 = $row.image_sha256.ToLowerInvariant()
        MaskSha256 = $row.mask_sha256.ToLowerInvariant()
        OriginalSplit = $row.split
        Plane = "unknown"
    }
    $candidates.Add($candidate)
    $oldCandidates.Add($candidate)
}

$briscRoot = Join-Path $ProjectRoot "DATASET\brisc2025"
$briscManifestPath = Join-Path $briscRoot "manifest.csv"
$briscRows = @(Import-Csv -LiteralPath $briscManifestPath)
$briscMasksByStem = @{}
foreach ($row in $briscRows | Where-Object { $_.task -eq "segmentation" -and $_.is_mask -eq "True" }) {
    $briscMasksByStem[[IO.Path]::GetFileNameWithoutExtension($row.filename)] = $row
}
$classMap = @{ gl = "Glioma"; me = "Meningioma"; pi = "Pituitary tumor" }
foreach ($row in $briscRows | Where-Object { $_.task -eq "segmentation" -and $_.is_mask -eq "False" }) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($row.filename)
    $maskRow = $briscMasksByStem[$stem]
    if ($null -eq $maskRow) {
        throw "Missing BRISC mask manifest row for $($row.filename)"
    }
    $imageAbsolute = Join-Path $briscRoot $row.relative_path
    $maskAbsolute = Join-Path $briscRoot $maskRow.relative_path
    if (-not (Test-Path -LiteralPath $imageAbsolute -PathType Leaf) -or
        -not (Test-Path -LiteralPath $maskAbsolute -PathType Leaf)) {
        throw "Missing BRISC pair: $imageAbsolute, $maskAbsolute"
    }
    $tumorType = $classMap[$row.tumor_code]
    if ([string]::IsNullOrWhiteSpace($tumorType)) {
        throw "Unknown BRISC tumor code: $($row.tumor_code)"
    }
    $candidate = [pscustomobject]@{
        Index = $candidates.Count
        CandidateId = "brisc:$stem"
        SourceDataset = "brisc2025"
        TumorType = $tumorType
        ImageAbsolute = $imageAbsolute
        MaskAbsolute = $maskAbsolute
        SourceImagePath = "DATASET/brisc2025/$($row.relative_path -replace '\\', '/')"
        SourceMaskPath = "DATASET/brisc2025/$($maskRow.relative_path -replace '\\', '/')"
        ImageSha256 = $row.sha256.ToLowerInvariant()
        MaskSha256 = $maskRow.sha256.ToLowerInvariant()
        OriginalSplit = $row.split
        Plane = $row.plane_label
    }
    $candidates.Add($candidate)
    $briscCandidates.Add($candidate)
}
Write-Output "Loaded $($oldCandidates.Count) original and $($briscCandidates.Count) BRISC segmentation pairs"

Write-Output "[2/7] Detecting exact and cross-dataset perceptual duplicates"
$script:Parent = [int[]](0..($candidates.Count - 1))
function Find-Root([int]$Node) {
    $root = $Node
    while ($script:Parent[$root] -ne $root) {
        $root = $script:Parent[$root]
    }
    while ($script:Parent[$Node] -ne $Node) {
        $next = $script:Parent[$Node]
        $script:Parent[$Node] = $root
        $Node = $next
    }
    return $root
}
function Union-Nodes([int]$First, [int]$Second) {
    $firstRoot = Find-Root $First
    $secondRoot = Find-Root $Second
    if ($firstRoot -ne $secondRoot) {
        if ($firstRoot -lt $secondRoot) {
            $script:Parent[$secondRoot] = $firstRoot
        }
        else {
            $script:Parent[$firstRoot] = $secondRoot
        }
    }
}

$exactGroups = @($candidates | Group-Object ImageSha256 | Where-Object Count -gt 1)
foreach ($group in $exactGroups) {
    $anchor = $group.Group[0].Index
    foreach ($member in $group.Group | Select-Object -Skip 1) {
        Union-Nodes $anchor $member.Index
    }
}

$oldPaths = [string[]]@($oldCandidates | ForEach-Object ImageAbsolute)
$briscPaths = [string[]]@($briscCandidates | ForEach-Object ImageAbsolute)
$nearMatchLines = [V2ImageTools]::FindCrossDatasetNearDuplicates(
    $oldPaths,
    $briscPaths,
    $CrossDatasetHammingDistance,
    $CrossDatasetCorrelation
)
$nearMatches = [Collections.Generic.List[object]]::new()
foreach ($line in $nearMatchLines) {
    $parts = $line -split ','
    $left = $oldCandidates[[int]$parts[0]]
    $right = $briscCandidates[[int]$parts[1]]
    Union-Nodes $left.Index $right.Index
    $nearMatches.Add([pscustomobject]@{
        FirstIndex = $left.Index
        SecondIndex = $right.Index
        HammingDistance = [int]$parts[2]
        Correlation = [double]::Parse($parts[3], [Globalization.CultureInfo]::InvariantCulture)
    })
}
Write-Output "Exact duplicate groups: $($exactGroups.Count); cross-dataset near-duplicate pairs: $($nearMatches.Count)"

Write-Output "[3/7] Resolving duplicate groups and quarantining conflicts"
$componentRows = @(
    foreach ($candidate in $candidates) {
        [pscustomobject]@{ Root = Find-Root $candidate.Index; Candidate = $candidate }
    }
)
$components = @($componentRows | Group-Object Root)
$kept = [Collections.Generic.List[object]]::new()
$quarantined = [Collections.Generic.List[object]]::new()

function Add-Quarantine([object]$Candidate, [string]$GroupId, [string]$Reason, [double]$MinimumMaskIoU) {
    $quarantined.Add([pscustomobject]@{
        candidate_id = $Candidate.CandidateId
        source_dataset = $Candidate.SourceDataset
        tumor_type = $Candidate.TumorType
        source_image_path = $Candidate.SourceImagePath
        source_mask_path = $Candidate.SourceMaskPath
        original_split = $Candidate.OriginalSplit
        plane = $Candidate.Plane
        image_sha256 = $Candidate.ImageSha256
        mask_sha256 = $Candidate.MaskSha256
        duplicate_group = $GroupId
        reason = $Reason
        minimum_mask_iou = if ($MinimumMaskIoU -lt 0) { "" } else { $MinimumMaskIoU.ToString("F6", [Globalization.CultureInfo]::InvariantCulture) }
    })
}

foreach ($component in $components) {
    $members = @($component.Group | ForEach-Object Candidate)
    $groupPayload = ($members.CandidateId | Sort-Object) -join "`n"
    $groupId = "dedup_" + ([V2ImageTools]::Sha256Text($groupPayload).Substring(0, 16))
    if ($members.Count -eq 1) {
        $members[0] | Add-Member -NotePropertyName DuplicateGroup -NotePropertyValue $groupId -Force
        $members[0] | Add-Member -NotePropertyName MinimumMaskIoU -NotePropertyValue 1.0 -Force
        $kept.Add($members[0])
        continue
    }

    $classes = @($members.TumorType | Select-Object -Unique)
    if ($classes.Count -gt 1) {
        foreach ($member in $members) {
            Add-Quarantine $member $groupId "label_conflict" -1
        }
        continue
    }

    $minimumIoU = 1.0
    $maskCompareFailed = $false
    for ($first = 0; $first -lt $members.Count; $first++) {
        for ($second = $first + 1; $second -lt $members.Count; $second++) {
            try {
                $iou = [V2ImageTools]::NormalizedMaskIoU(
                    $members[$first].MaskAbsolute,
                    $members[$second].MaskAbsolute,
                    $TargetSize
                )
                if ($iou -lt $minimumIoU) { $minimumIoU = $iou }
            }
            catch {
                $maskCompareFailed = $true
            }
        }
    }
    if ($maskCompareFailed -or $minimumIoU -lt $DuplicateMaskIoU) {
        $reason = if ($maskCompareFailed) { "mask_compare_error" } else { "mask_conflict" }
        foreach ($member in $members) {
            Add-Quarantine $member $groupId $reason $minimumIoU
        }
        continue
    }

    $canonical = @(
        $members | Sort-Object @{ Expression = { if ($_.SourceDataset -eq "original") { 0 } else { 1 } } }, SourceImagePath
    )[0]
    $canonical | Add-Member -NotePropertyName DuplicateGroup -NotePropertyValue $groupId -Force
    $canonical | Add-Member -NotePropertyName MinimumMaskIoU -NotePropertyValue $minimumIoU -Force
    $kept.Add($canonical)
    foreach ($member in $members | Where-Object Index -ne $canonical.Index) {
        Add-Quarantine $member $groupId "duplicate_redundant" $minimumIoU
    }
}
Write-Output "Selected $($kept.Count) candidates; quarantined $($quarantined.Count) candidates"

Write-Output "[4/7] Converting selected pairs to compatible 512x512 grayscale PNG"
$outputRecords = [Collections.Generic.List[object]]::new()
$conversionFailures = 0
$convertedCount = 0
foreach ($classGroup in $kept | Group-Object TumorType | Sort-Object Name) {
    $newId = 0
    $ordered = @(
        $classGroup.Group |
            Sort-Object @{ Expression = { if ($_.SourceDataset -eq "original") { 0 } else { 1 } } }, SourceImagePath
    )
    foreach ($candidate in $ordered) {
        $newId++
        $classDirectory = Join-Path $outputRootAbsolute $candidate.TumorType
        $imageName = "enh_$newId.png"
        $maskName = "enh_$($newId)_mask.png"
        $outputImage = Join-Path $classDirectory $imageName
        $outputMask = Join-Path $classDirectory $maskName
        try {
            $stats = [V2ImageTools]::ConvertPair(
                $candidate.ImageAbsolute,
                $candidate.MaskAbsolute,
                $outputImage,
                $outputMask,
                $TargetSize
            )
            $imageRelative = "$($candidate.TumorType)/$imageName"
            $maskRelative = "$($candidate.TumorType)/$maskName"
            $sampleId = "$($candidate.TumorType)__enh_$newId"
            $outputRecords.Add([pscustomobject]@{
                CandidateId = $candidate.CandidateId
                SampleId = $sampleId
                SourceId = $newId
                TumorType = $candidate.TumorType
                SourceDataset = $candidate.SourceDataset
                SourceImagePath = $candidate.SourceImagePath
                SourceMaskPath = $candidate.SourceMaskPath
                OriginalSplit = $candidate.OriginalSplit
                Plane = $candidate.Plane
                DuplicateGroup = $candidate.DuplicateGroup
                ImagePath = $imageRelative
                MaskPath = $maskRelative
                ImageAbsolute = $outputImage
                MaskAbsolute = $outputMask
                RawImageSha256 = $candidate.ImageSha256
                RawMaskSha256 = $candidate.MaskSha256
                ImageSha256 = [V2ImageTools]::Sha256File($outputImage)
                MaskSha256 = [V2ImageTools]::Sha256File($outputMask)
                ForegroundFraction = $stats.ForegroundFraction
            })
            $convertedCount++
            if ($convertedCount % 250 -eq 0) {
                Write-Output "Converted $convertedCount / $($kept.Count)"
            }
        }
        catch {
            $conversionFailures++
            if (Test-Path -LiteralPath $outputImage) { Remove-Item -LiteralPath $outputImage -Force }
            if (Test-Path -LiteralPath $outputMask) { Remove-Item -LiteralPath $outputMask -Force }
            Add-Quarantine $candidate $candidate.DuplicateGroup ("conversion_error: " + $_.Exception.Message) -1
        }
    }
}
if ($conversionFailures -gt 0) {
    Write-Warning "$conversionFailures selected pairs failed conversion and were quarantined"
}
if ($outputRecords.Count -eq 0) {
    throw "No V2 samples were generated"
}

# Grayscale conversion can collapse distinct RGB encodings into identical L images. Resolve those
# groups with the same conservative mask-IoU rule before writing manifests or assigning splits.
$postNormalizationGroups = @($outputRecords | Group-Object ImageSha256 | Where-Object Count -gt 1)
$postNormalizationRemoved = New-Object 'Collections.Generic.HashSet[string]'
foreach ($group in $postNormalizationGroups) {
    $members = @($group.Group)
    $groupPayload = ($members.CandidateId | Sort-Object) -join "`n"
    $groupId = "normalized_" + ([V2ImageTools]::Sha256Text($groupPayload).Substring(0, 16))
    $classes = @($members.TumorType | Select-Object -Unique)
    $minimumIoU = 1.0
    if ($classes.Count -le 1) {
        for ($first = 0; $first -lt $members.Count; $first++) {
            for ($second = $first + 1; $second -lt $members.Count; $second++) {
                $iou = [V2ImageTools]::NormalizedMaskIoU(
                    $members[$first].MaskAbsolute,
                    $members[$second].MaskAbsolute,
                    $TargetSize
                )
                if ($iou -lt $minimumIoU) { $minimumIoU = $iou }
            }
        }
    }
    $canonical = $null
    if ($classes.Count -le 1 -and $minimumIoU -ge $DuplicateMaskIoU) {
        $canonical = @(
            $members | Sort-Object @{ Expression = { if ($_.SourceDataset -eq "original") { 0 } else { 1 } } }, SourceImagePath
        )[0]
        $canonical.DuplicateGroup = $groupId
    }
    foreach ($member in $members) {
        if ($null -ne $canonical -and $member.SampleId -eq $canonical.SampleId) {
            continue
        }
        [void]$postNormalizationRemoved.Add($member.SampleId)
        $reason = if ($classes.Count -gt 1) {
            "post_normalization_label_conflict"
        }
        elseif ($minimumIoU -lt $DuplicateMaskIoU) {
            "post_normalization_mask_conflict"
        }
        else {
            "post_normalization_duplicate_redundant"
        }
        $pseudoCandidate = [pscustomobject]@{
            CandidateId = $member.CandidateId
            SourceDataset = $member.SourceDataset
            TumorType = $member.TumorType
            SourceImagePath = $member.SourceImagePath
            SourceMaskPath = $member.SourceMaskPath
            OriginalSplit = $member.OriginalSplit
            Plane = $member.Plane
            ImageSha256 = $member.RawImageSha256
            MaskSha256 = $member.RawMaskSha256
        }
        Add-Quarantine $pseudoCandidate $groupId $reason $minimumIoU
        if (Test-Path -LiteralPath $member.ImageAbsolute) { Remove-Item -LiteralPath $member.ImageAbsolute -Force }
        if (Test-Path -LiteralPath $member.MaskAbsolute) { Remove-Item -LiteralPath $member.MaskAbsolute -Force }
    }
}
if ($postNormalizationRemoved.Count -gt 0) {
    $survivors = [Collections.Generic.List[object]]::new()
    foreach ($record in $outputRecords) {
        if (-not $postNormalizationRemoved.Contains($record.SampleId)) {
            $survivors.Add($record)
        }
    }
    $outputRecords = $survivors
}
Write-Output "Post-normalization duplicate groups: $($postNormalizationGroups.Count); final selected pairs: $($outputRecords.Count)"

$cleanRows = @(
    $outputRecords | ForEach-Object {
        [pscustomobject]@{
            sample_id = $_.SampleId
            source_id = $_.SourceId
            tumor_type = $_.TumorType
            source_dataset = $_.SourceDataset
            source_image_path = $_.SourceImagePath
            source_mask_path = $_.SourceMaskPath
            original_split = $_.OriginalSplit
            plane = $_.Plane
            duplicate_group = $_.DuplicateGroup
            image_path = $_.ImagePath
            mask_path = $_.MaskPath
            image_sha256 = $_.ImageSha256
            mask_sha256 = $_.MaskSha256
            foreground_fraction = $_.ForegroundFraction.ToString("F8", [Globalization.CultureInfo]::InvariantCulture)
        }
    }
)
$cleanRows | Export-Csv -LiteralPath $cleanManifestAbsolute -NoTypeInformation -Encoding UTF8
$quarantined | Export-Csv -LiteralPath $quarantineManifestAbsolute -NoTypeInformation -Encoding UTF8

Write-Output "[5/7] Building similarity groups used only for leakage-safe splitting"
$script:SplitParent = [int[]](0..($outputRecords.Count - 1))
function Find-SplitRoot([int]$Node) {
    $root = $Node
    while ($script:SplitParent[$root] -ne $root) { $root = $script:SplitParent[$root] }
    while ($script:SplitParent[$Node] -ne $Node) {
        $next = $script:SplitParent[$Node]
        $script:SplitParent[$Node] = $root
        $Node = $next
    }
    return $root
}
function Union-SplitNodes([int]$First, [int]$Second) {
    $firstRoot = Find-SplitRoot $First
    $secondRoot = Find-SplitRoot $Second
    if ($firstRoot -ne $secondRoot) {
        if ($firstRoot -lt $secondRoot) { $script:SplitParent[$secondRoot] = $firstRoot }
        else { $script:SplitParent[$firstRoot] = $secondRoot }
    }
}

$recordIndex = @{}
for ($index = 0; $index -lt $outputRecords.Count; $index++) {
    $recordIndex[$outputRecords[$index].SampleId] = $index
}
$splitSimilarityPairs = [Collections.Generic.List[object]]::new()
foreach ($stratum in $outputRecords | Group-Object TumorType, SourceDataset) {
    $members = @($stratum.Group)
    $paths = [string[]]@($members | ForEach-Object ImageAbsolute)
    $lines = [V2ImageTools]::FindWithinDatasetNearDuplicates(
        $paths,
        $SplitGroupHammingDistance,
        $SplitGroupCorrelation
    )
    foreach ($line in $lines) {
        $parts = $line -split ','
        $firstRecord = $members[[int]$parts[0]]
        $secondRecord = $members[[int]$parts[1]]
        $firstIndex = $recordIndex[$firstRecord.SampleId]
        $secondIndex = $recordIndex[$secondRecord.SampleId]
        Union-SplitNodes $firstIndex $secondIndex
        $splitSimilarityPairs.Add([pscustomobject]@{
            first = $firstRecord.SampleId
            second = $secondRecord.SampleId
            hamming_distance = [int]$parts[2]
            correlation = [double]::Parse($parts[3], [Globalization.CultureInfo]::InvariantCulture)
        })
    }
}
for ($index = 0; $index -lt $outputRecords.Count; $index++) {
    $root = Find-SplitRoot $index
    $outputRecords[$index] | Add-Member -NotePropertyName SplitGroup -NotePropertyValue ("similarity_" + $root) -Force
}
Write-Output "Found $($splitSimilarityPairs.Count) high-similarity same-source pairs; they will be kept in the same split"

Write-Output "[6/7] Assigning source/class-stratified 70/15/15 splits"
$assignment = @{}
foreach ($stratum in $outputRecords | Group-Object TumorType, SourceDataset) {
    $members = @($stratum.Group)
    $groups = @($members | Group-Object SplitGroup)
    $total = $members.Count
    $target = @{ train = 0.70 * $total; val = 0.15 * $total; test = 0.15 * $total }
    $counts = @{ train = 0; val = 0; test = 0 }
    $orderedGroups = @(
        $groups |
            ForEach-Object {
                [pscustomobject]@{
                    Group = $_
                    Size = $_.Count
                    Tie = [V2ImageTools]::Sha256Text("$Seed|$($stratum.Name)|$($_.Name)")
                }
            } |
            Sort-Object @{ Expression = { -$_.Size } }, Tie
    )
    foreach ($groupInfo in $orderedGroups) {
        $bestSplit = @("train", "val", "test") |
            Sort-Object @{ Expression = { -(($target[$_] - $counts[$_]) / [Math]::Max($target[$_], 1.0)) } }, @{ Expression = { [V2ImageTools]::Sha256Text("$Seed|$($groupInfo.Group.Name)|$_") } } |
            Select-Object -First 1
        foreach ($member in $groupInfo.Group.Group) {
            $assignment[$member.SampleId] = $bestSplit
        }
        $counts[$bestSplit] += $groupInfo.Size
    }
}

$splitOrder = @{ train = 0; val = 1; test = 2 }
$splitRows = @(
    $outputRecords |
        ForEach-Object {
            [pscustomobject]@{
                sample_id = $_.SampleId
                source_id = $_.SourceId
                tumor_type = $_.TumorType
                group_id = $_.SplitGroup
                split = $assignment[$_.SampleId]
                image_path = $_.ImagePath
                mask_path = $_.MaskPath
                image_sha256 = $_.ImageSha256
                mask_sha256 = $_.MaskSha256
            }
        } |
        Sort-Object @{ Expression = { $splitOrder[$_.split] } }, tumor_type, source_id
)
$splitRows | Export-Csv -LiteralPath $splitManifestAbsolute -NoTypeInformation -Encoding UTF8

Write-Output "[7/7] Final validation and report"
$outputExactDuplicates = @($outputRecords | Group-Object ImageSha256 | Where-Object Count -gt 1)
if ($outputExactDuplicates.Count -gt 0) {
    throw "Output validation failed: $($outputExactDuplicates.Count) exact image duplicate groups remain"
}
$splitGroupLeaks = @(
    $splitRows |
        Group-Object group_id |
        Where-Object { @($_.Group.split | Select-Object -Unique).Count -gt 1 }
)
if ($splitGroupLeaks.Count -gt 0) {
    throw "Output validation failed: similarity groups leak across splits"
}
$splitCounts = @(
    $splitRows |
        Group-Object split |
        Sort-Object Name |
        ForEach-Object { [pscustomobject]@{ split = $_.Name; count = $_.Count } }
)
$splitClassSourceCounts = @(
    $splitRows |
        ForEach-Object {
            $source = ($outputRecords | Where-Object SampleId -eq $_.sample_id | Select-Object -First 1).SourceDataset
            [pscustomobject]@{ split = $_.split; tumor_type = $_.tumor_type; source_dataset = $source }
        } |
        Group-Object split, tumor_type, source_dataset |
        Sort-Object Name |
        ForEach-Object {
            [pscustomobject]@{
                split = $_.Group[0].split
                tumor_type = $_.Group[0].tumor_type
                source_dataset = $_.Group[0].source_dataset
                count = $_.Count
            }
        }
)
$manifestSha256 = [V2ImageTools]::Sha256File($splitManifestAbsolute)
$fingerprintPayload = @(
    $splitRows |
        Sort-Object sample_id |
        ForEach-Object { "$($_.sample_id)`0$($_.image_path)`0$($_.mask_path)`0$($_.image_sha256)`0$($_.mask_sha256)" }
) -join "`n"
$datasetFingerprint = [V2ImageTools]::Sha256Text($fingerprintPayload)

$metadata = [ordered]@{
    schema_version = 2
    created_at_utc = [DateTime]::UtcNow.ToString("o", [Globalization.CultureInfo]::InvariantCulture)
    seed = $Seed
    ratios = [ordered]@{ train = 0.70; val = 0.15; test = 0.15 }
    split_level = "source_class_similarity_group"
    group_csv = $null
    num_samples = $splitRows.Count
    counts = $splitCounts
    per_split_class_source_counts = $splitClassSourceCounts
    dataset_fingerprint = $datasetFingerprint
    manifest_sha256 = $manifestSha256
    mask_binarization = "mask >= 128 -> {0,255}"
    image_preprocessing = "grayscale; aspect-ratio-preserving resize; centered black padding to 512x512"
    known_limitation = "Source patient IDs are unavailable; similarity grouping reduces but cannot prove patient-level independence."
}
Write-Utf8Json $metadata $splitMetadataAbsolute

$report = [ordered]@{
    schema_version = 1
    created_at_utc = [DateTime]::UtcNow.ToString("o", [Globalization.CultureInfo]::InvariantCulture)
    policy = "safe"
    inputs = [ordered]@{
        original_pairs = $oldCandidates.Count
        brisc_pairs = $briscCandidates.Count
        total_pairs = $candidates.Count
        by_class_and_source = Get-ClassSourceCounts @($candidates)
    }
    duplicate_detection = [ordered]@{
        exact_hash_groups = $exactGroups.Count
        cross_dataset_near_duplicate_pairs = $nearMatches.Count
        post_normalization_exact_groups = $postNormalizationGroups.Count
        cross_dataset_max_dhash_distance = $CrossDatasetHammingDistance
        cross_dataset_minimum_pixel_correlation = $CrossDatasetCorrelation
        duplicate_mask_minimum_iou = $DuplicateMaskIoU
    }
    cleaning = [ordered]@{
        generated_pairs = $outputRecords.Count
        quarantined_candidates = $quarantined.Count
        conversion_failures = $conversionFailures
        output_by_class_and_source = Get-ClassSourceCounts @($outputRecords)
        quarantine_by_reason = @(
            $quarantined | Group-Object reason | Sort-Object Name | ForEach-Object {
                [pscustomobject]@{ reason = $_.Name; count = $_.Count }
            }
        )
    }
    output_contract = [ordered]@{
        root = To-ProjectRelativePath $outputRootAbsolute
        format = "PNG"
        image_mode = "L / 8-bit indexed grayscale"
        mask_values = @(0, 255)
        width = $TargetSize
        height = $TargetSize
        naming = "<tumor_type>/enh_<id>.png and enh_<id>_mask.png"
        exact_duplicate_groups_remaining = $outputExactDuplicates.Count
    }
    split = [ordered]@{
        manifest = To-ProjectRelativePath $splitManifestAbsolute
        metadata = To-ProjectRelativePath $splitMetadataAbsolute
        same_source_similarity_pairs_grouped = $splitSimilarityPairs.Count
        maximum_dhash_distance = $SplitGroupHammingDistance
        minimum_pixel_correlation = $SplitGroupCorrelation
        group_leaks = $splitGroupLeaks.Count
        counts = $splitCounts
        per_split_class_source_counts = $splitClassSourceCounts
    }
    artifacts = [ordered]@{
        clean_manifest = To-ProjectRelativePath $cleanManifestAbsolute
        quarantine_manifest = To-ProjectRelativePath $quarantineManifestAbsolute
        report = To-ProjectRelativePath $reportAbsolute
    }
    limitations = @(
        "Neither source exposes reliable patient IDs.",
        "The split is leakage-controlled at exact/perceptual similarity-group level, not proven patient level.",
        "Quarantined label and mask conflicts require manual medical review before recovery."
    )
}
Write-Utf8Json $report $reportAbsolute

Write-Output "Generated $($outputRecords.Count) validated V2 pairs at $outputRootAbsolute"
Write-Output "Quarantined $($quarantined.Count) source candidates"
Write-Output "Split counts: $((($splitCounts | ForEach-Object { $_.split + '=' + $_.count }) -join ', '))"
Write-Output "Report: $reportAbsolute"
