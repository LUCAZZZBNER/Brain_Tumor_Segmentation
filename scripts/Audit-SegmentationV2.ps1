[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$DatasetRoot = "DATASET/Segmentation_v2",
    [string]$CleanManifest = "DATASET/Segmentation_v2_manifest.csv",
    [string]$SplitManifest = "splits/segmentation_v2_seed42.csv",
    [string]$OutputDirectory = "reports/segmentation_v2_audit",
    [int]$MaximumHammingDistance = 12,
    [double]$MinimumCorrelation = 0.95
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Value))
}
function Relative-Path([string]$Absolute) {
    $rootUri = [Uri](($ProjectRoot.TrimEnd('\') + '\'))
    return [Uri]::UnescapeDataString($rootUri.MakeRelativeUri([Uri]$Absolute).ToString())
}
function New-Issue([string]$Severity, [string]$Code, [int]$Count, [string]$Detail) {
    [pscustomobject]@{ severity = $Severity; code = $Code; count = $Count; detail = $Detail }
}

$datasetAbsolute = Resolve-ProjectPath $DatasetRoot
$cleanAbsolute = Resolve-ProjectPath $CleanManifest
$splitAbsolute = Resolve-ProjectPath $SplitManifest
$outputAbsolute = Resolve-ProjectPath $OutputDirectory
New-Item -ItemType Directory -Path $outputAbsolute -Force | Out-Null

Add-Type -AssemblyName System.Drawing
Add-Type -Path (Join-Path $PSScriptRoot "SegmentationV2AuditTools.cs") -ReferencedAssemblies System.Drawing

Write-Output "[1/5] Loading and cross-checking manifests"
$cleanRows = @(Import-Csv -LiteralPath $cleanAbsolute)
$splitRows = @(Import-Csv -LiteralPath $splitAbsolute)
$cleanById = @{}
foreach ($row in $cleanRows) {
    if ($cleanById.ContainsKey($row.sample_id)) { throw "Duplicate sample_id in clean manifest: $($row.sample_id)" }
    $cleanById[$row.sample_id] = $row
}
$splitById = @{}
foreach ($row in $splitRows) {
    if ($splitById.ContainsKey($row.sample_id)) { throw "Duplicate sample_id in split manifest: $($row.sample_id)" }
    $splitById[$row.sample_id] = $row
}

$auditRows = [Collections.Generic.List[object]]::new()
foreach ($row in $splitRows) {
    $clean = $cleanById[$row.sample_id]
    $imageAbsolute = Join-Path $datasetAbsolute ($row.image_path -replace '/', '\')
    $maskAbsolute = Join-Path $datasetAbsolute ($row.mask_path -replace '/', '\')
    $auditRows.Add([pscustomobject]@{
        index = $auditRows.Count; sample_id = $row.sample_id; tumor_type = $row.tumor_type
        source_dataset = if ($null -eq $clean) { "" } else { $clean.source_dataset }
        original_split = if ($null -eq $clean) { "" } else { $clean.original_split }
        split = $row.split; group_id = $row.group_id
        image_path = $row.image_path; mask_path = $row.mask_path
        image_absolute = $imageAbsolute; mask_absolute = $maskAbsolute
        manifest_image_sha256 = $row.image_sha256.ToLowerInvariant()
        manifest_mask_sha256 = $row.mask_sha256.ToLowerInvariant()
        image_exists = Test-Path -LiteralPath $imageAbsolute -PathType Leaf
        mask_exists = Test-Path -LiteralPath $maskAbsolute -PathType Leaf
    })
}

$issues = [Collections.Generic.List[object]]::new()
$cleanOnly = @($cleanRows | Where-Object { -not $splitById.ContainsKey($_.sample_id) })
$splitOnly = @($splitRows | Where-Object { -not $cleanById.ContainsKey($_.sample_id) })
if ($cleanOnly.Count) { $issues.Add((New-Issue high clean_samples_missing_from_split $cleanOnly.Count "Clean manifest rows absent from split manifest.")) }
if ($splitOnly.Count) { $issues.Add((New-Issue high split_samples_missing_from_clean $splitOnly.Count "Split rows absent from provenance manifest.")) }
$missingPairs = @($auditRows | Where-Object { -not $_.image_exists -or -not $_.mask_exists })
if ($missingPairs.Count) { $issues.Add((New-Issue critical missing_files $missingPairs.Count "Manifest image or mask does not exist.")) }
if ($missingPairs.Count) { throw "Cannot inspect pixels because $($missingPairs.Count) pairs have missing files." }

Write-Output "[2/5] Decoding every image and mask"
$inspectionLines = [SegmentationV2AuditTools]::InspectPairs(
    [string[]]@($auditRows.image_absolute), [string[]]@($auditRows.mask_absolute))
foreach ($line in $inspectionLines) {
    $p = $line -split '\|'
    $row = $auditRows[[int]$p[0]]
    $row | Add-Member NoteProperty image_width ([int]$p[1])
    $row | Add-Member NoteProperty image_height ([int]$p[2])
    $row | Add-Member NoteProperty image_bpp ([int]$p[3])
    $row | Add-Member NoteProperty mask_width ([int]$p[4])
    $row | Add-Member NoteProperty mask_height ([int]$p[5])
    $row | Add-Member NoteProperty mask_bpp ([int]$p[6])
    $row | Add-Member NoteProperty image_pixel_sha256 $p[7]
    $row | Add-Member NoteProperty mask_pixel_sha256 $p[8]
    $row | Add-Member NoteProperty image_distinct_values ([int]$p[9])
    $row | Add-Member NoteProperty image_min ([int]$p[10])
    $row | Add-Member NoteProperty image_max ([int]$p[11])
    $row | Add-Member NoteProperty image_mean ([double]::Parse($p[12], [Globalization.CultureInfo]::InvariantCulture))
    $row | Add-Member NoteProperty image_stddev ([double]::Parse($p[13], [Globalization.CultureInfo]::InvariantCulture))
    $row | Add-Member NoteProperty mask_distinct_values ([int]$p[14])
    $row | Add-Member NoteProperty mask_binary ([int]$p[15] -eq 1)
    $row | Add-Member NoteProperty foreground_pixels ([long]$p[16])
    $row | Add-Member NoteProperty foreground_fraction ([long]$p[16] / ([int]$p[4] * [int]$p[5]))
    $row | Add-Member NoteProperty mask_min_x ([int]$p[17])
    $row | Add-Member NoteProperty mask_min_y ([int]$p[18])
    $row | Add-Member NoteProperty mask_max_x ([int]$p[19])
    $row | Add-Member NoteProperty mask_max_y ([int]$p[20])
    $row | Add-Member NoteProperty actual_image_sha256 ((Get-FileHash -LiteralPath $row.image_absolute -Algorithm SHA256).Hash.ToLowerInvariant())
    $row | Add-Member NoteProperty actual_mask_sha256 ((Get-FileHash -LiteralPath $row.mask_absolute -Algorithm SHA256).Hash.ToLowerInvariant())
}

$badHash = @($auditRows | Where-Object { $_.actual_image_sha256 -ne $_.manifest_image_sha256 -or $_.actual_mask_sha256 -ne $_.manifest_mask_sha256 })
$badDimensions = @($auditRows | Where-Object { $_.image_width -ne 512 -or $_.image_height -ne 512 -or $_.mask_width -ne 512 -or $_.mask_height -ne 512 })
$badBpp = @($auditRows | Where-Object { $_.image_bpp -ne 8 -or $_.mask_bpp -ne 8 })
$badMasks = @($auditRows | Where-Object { -not $_.mask_binary -or $_.foreground_pixels -eq 0 -or $_.foreground_pixels -eq 262144 })
$lowInformation = @($auditRows | Where-Object { $_.image_distinct_values -lt 16 -or $_.image_stddev -lt 5 })
if ($badHash.Count) { $issues.Add((New-Issue critical content_hash_mismatch $badHash.Count "File bytes differ from immutable split manifest.")) }
if ($badDimensions.Count) { $issues.Add((New-Issue high bad_dimensions $badDimensions.Count "Image or mask is not 512x512.")) }
if ($badBpp.Count) { $issues.Add((New-Issue medium unexpected_bit_depth $badBpp.Count "Image or mask is not decoded as 8-bit.")) }
if ($badMasks.Count) { $issues.Add((New-Issue high invalid_masks $badMasks.Count "Mask is non-binary, empty, or full.")) }
if ($lowInformation.Count) { $issues.Add((New-Issue medium low_information_images $lowInformation.Count "Image has fewer than 16 values or standard deviation below 5.")) }

Write-Output "[3/5] Checking exact decoded-content duplicates"
$imageDuplicateGroups = @($auditRows | Group-Object image_pixel_sha256 | Where-Object Count -gt 1)
$maskDuplicateGroups = @($auditRows | Group-Object mask_pixel_sha256 | Where-Object Count -gt 1)
$pairDuplicateGroups = @($auditRows | Group-Object { "$($_.image_pixel_sha256)|$($_.mask_pixel_sha256)" } | Where-Object Count -gt 1)
$crossSplitImageGroups = @($imageDuplicateGroups | Where-Object { @($_.Group.split | Sort-Object -Unique).Count -gt 1 })
$crossSplitMaskGroups = @($maskDuplicateGroups | Where-Object { @($_.Group.split | Sort-Object -Unique).Count -gt 1 })
$crossSplitPairGroups = @($pairDuplicateGroups | Where-Object { @($_.Group.split | Sort-Object -Unique).Count -gt 1 })
if ($imageDuplicateGroups.Count) { $issues.Add((New-Issue high exact_image_duplicate_groups $imageDuplicateGroups.Count "Decoded image pixels repeat within the dataset.")) }
if ($crossSplitImageGroups.Count) { $issues.Add((New-Issue critical cross_split_exact_image_groups $crossSplitImageGroups.Count "Exact decoded image pixels occur in multiple splits.")) }
if ($crossSplitPairGroups.Count) { $issues.Add((New-Issue critical cross_split_exact_pair_groups $crossSplitPairGroups.Count "Exact image-mask pairs occur in multiple splits.")) }
if ($maskDuplicateGroups.Count) { $issues.Add((New-Issue medium exact_mask_duplicate_groups $maskDuplicateGroups.Count "Decoded non-empty mask pixels repeat; inspect whether annotations were copied.")) }
if ($crossSplitMaskGroups.Count) { $issues.Add((New-Issue medium cross_split_exact_mask_groups $crossSplitMaskGroups.Count "Exact masks occur in multiple splits; mask-only repetition is not automatically leakage.")) }

$exactRows = @(
    foreach ($group in $imageDuplicateGroups) { foreach ($row in $group.Group) { [pscustomobject]@{ kind='image'; hash=$group.Name; group_size=$group.Count; sample_id=$row.sample_id; split=$row.split; tumor_type=$row.tumor_type; image_path=$row.image_path; mask_path=$row.mask_path } } }
    foreach ($group in $maskDuplicateGroups) { foreach ($row in $group.Group) { [pscustomobject]@{ kind='mask'; hash=$group.Name; group_size=$group.Count; sample_id=$row.sample_id; split=$row.split; tumor_type=$row.tumor_type; image_path=$row.image_path; mask_path=$row.mask_path } } }
)

Write-Output "[4/5] Searching all near-duplicates (including 8 orientations)"
$uniqueComparisonLabels = [string[]]@(0..($auditRows.Count - 1) | ForEach-Object { "sample_$_" })
$nearLines = [SegmentationV2AuditTools]::FindCrossSplitNearDuplicates(
    [string[]]@($auditRows.image_absolute), [string[]]@($auditRows.mask_absolute),
    $uniqueComparisonLabels, $MaximumHammingDistance, $MinimumCorrelation)
$nearRows = @(
    foreach ($line in $nearLines) {
        $p = $line -split '\|'; $first = $auditRows[[int]$p[0]]; $second = $auditRows[[int]$p[1]]
        [pscustomobject]@{
            first_index=[int]$p[0]; second_index=[int]$p[1]
            first_sample_id=$first.sample_id; first_split=$first.split; first_class=$first.tumor_type; first_source=$first.source_dataset; first_image_path=$first.image_path
            second_sample_id=$second.sample_id; second_split=$second.split; second_class=$second.tumor_type; second_source=$second.source_dataset; second_image_path=$second.image_path
            hamming_distance=[int]$p[2]; pixel_correlation=[double]::Parse($p[3], [Globalization.CultureInfo]::InvariantCulture)
            mask_iou=[double]::Parse($p[4], [Globalization.CultureInfo]::InvariantCulture); transform=$p[5]
            strong=([int]$p[2] -le 8 -and [double]::Parse($p[3], [Globalization.CultureInfo]::InvariantCulture) -ge 0.98)
        }
    }
)
$verificationLines = [SegmentationV2AuditTools]::VerifyTransformedPairs(
    [string[]]@($nearRows | ForEach-Object { $auditRows[$_.first_index].image_absolute }),
    [string[]]@($nearRows | ForEach-Object { $auditRows[$_.first_index].mask_absolute }),
    [string[]]@($nearRows | ForEach-Object { $auditRows[$_.second_index].image_absolute }),
    [string[]]@($nearRows | ForEach-Object { $auditRows[$_.second_index].mask_absolute }),
    [string[]]@($nearRows.transform))
foreach ($line in $verificationLines) {
    $p = $line -split '\|'; $row = $nearRows[[int]$p[0]]
    $row | Add-Member NoteProperty full_pixel_correlation ([double]::Parse($p[1], [Globalization.CultureInfo]::InvariantCulture))
    $row | Add-Member NoteProperty full_image_mae ([double]::Parse($p[2], [Globalization.CultureInfo]::InvariantCulture))
    $row | Add-Member NoteProperty full_image_rmse ([double]::Parse($p[3], [Globalization.CultureInfo]::InvariantCulture))
    $row | Add-Member NoteProperty full_image_psnr $p[4]
    $row | Add-Member NoteProperty full_image_mismatch_pixels ([long]$p[5])
    $row | Add-Member NoteProperty full_mask_iou ([double]::Parse($p[6], [Globalization.CultureInfo]::InvariantCulture))
    $row | Add-Member NoteProperty full_mask_mismatch_pixels ([long]$p[7])
    $row | Add-Member NoteProperty likely_duplicate ($row.full_pixel_correlation -ge 0.98 -and $row.full_mask_iou -ge 0.90)
}
$allLikelyNear = @($nearRows | Where-Object likely_duplicate)
$withinSplitLikely = @($allLikelyNear | Where-Object { $_.first_split -eq $_.second_split })
$likelyNear = @($allLikelyNear | Where-Object { $_.first_split -ne $_.second_split })
$strongNear = @($nearRows | Where-Object { $_.strong -and $_.first_split -ne $_.second_split })
$crossClassLikely = @($allLikelyNear | Where-Object { $_.first_class -ne $_.second_class })
$duplicateIds = @($allLikelyNear.first_sample_id + $allLikelyNear.second_sample_id | Sort-Object -Unique)
$script:AuditDuplicateParent = @{}
foreach ($sampleId in $duplicateIds) { $script:AuditDuplicateParent[$sampleId] = $sampleId }
function Find-AuditDuplicateRoot([string]$SampleId) {
    $root = $SampleId
    while ($script:AuditDuplicateParent[$root] -ne $root) { $root = $script:AuditDuplicateParent[$root] }
    while ($script:AuditDuplicateParent[$SampleId] -ne $SampleId) {
        $next = $script:AuditDuplicateParent[$SampleId]
        $script:AuditDuplicateParent[$SampleId] = $root
        $SampleId = $next
    }
    return $root
}
foreach ($pair in $allLikelyNear) {
    $firstRoot = Find-AuditDuplicateRoot $pair.first_sample_id
    $secondRoot = Find-AuditDuplicateRoot $pair.second_sample_id
    if ($firstRoot -ne $secondRoot) { $script:AuditDuplicateParent[$secondRoot] = $firstRoot }
}
$duplicateMembers = @($duplicateIds | ForEach-Object {
    [pscustomobject]@{ root=(Find-AuditDuplicateRoot $_); sample_id=$_; split=$splitById[$_].split; tumor_type=$splitById[$_].tumor_type }
})
$duplicateComponents = @($duplicateMembers | Group-Object root)
$leakingDuplicateComponents = @($duplicateComponents | Where-Object { @($_.Group.split | Sort-Object -Unique).Count -gt 1 })
$duplicateComponentRows = @(
    foreach ($component in $duplicateComponents) {
        $componentSplits = @($component.Group.split | Sort-Object -Unique) -join ','
        foreach ($member in $component.Group) {
            [pscustomobject]@{ component_id=$component.Name; component_size=$component.Count; crosses_splits=($componentSplits -match ','); splits=$componentSplits; sample_id=$member.sample_id; split=$member.split; tumor_type=$member.tumor_type }
        }
    }
)
if ($allLikelyNear.Count) { $issues.Add((New-Issue high all_likely_duplicate_pairs $allLikelyNear.Count "Full-resolution correlation>=0.98 and transformed mask IoU>=0.90.")) }
if ($strongNear.Count) { $issues.Add((New-Issue high cross_split_strong_near_pairs $strongNear.Count "dHash<=8 and correlation>=0.98 across splits; includes transformed comparisons.")) }
if ($likelyNear.Count) { $issues.Add((New-Issue critical cross_split_likely_duplicate_pairs $likelyNear.Count "Full-resolution correlation>=0.98 and transformed mask IoU>=0.90 across splits.")) }
if ($crossClassLikely.Count) { $issues.Add((New-Issue critical cross_class_likely_duplicate_pairs $crossClassLikely.Count "Near-identical image/mask pairs have conflicting tumor classes.")) }

Write-Output "[5/5] Writing reproducible audit artifacts"
$filesystemFiles = @(Get-ChildItem -LiteralPath $datasetAbsolute -Recurse -File)
$expectedPaths = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($row in $auditRows) { [void]$expectedPaths.Add($row.image_absolute); [void]$expectedPaths.Add($row.mask_absolute) }
$unreferenced = @($filesystemFiles | Where-Object { -not $expectedPaths.Contains($_.FullName) })
if ($unreferenced.Count) { $issues.Add((New-Issue medium unreferenced_dataset_files $unreferenced.Count "Files exist below dataset root but are absent from the split manifest.")) }

$groupLeaks = @($splitRows | Group-Object group_id | Where-Object { @($_.Group.split | Sort-Object -Unique).Count -gt 1 })
if ($groupLeaks.Count) { $issues.Add((New-Issue critical declared_similarity_group_leaks $groupLeaks.Count "A declared group_id occurs in multiple splits.")) }
$pathDuplicates = @($splitRows | Group-Object image_path | Where-Object Count -gt 1)
if ($pathDuplicates.Count) { $issues.Add((New-Issue critical duplicate_image_paths $pathDuplicates.Count "An image path is referenced by multiple samples.")) }

$transition = @($auditRows | Group-Object source_dataset, original_split, split | Sort-Object Name | ForEach-Object {
    [pscustomobject]@{ source_dataset=$_.Group[0].source_dataset; original_split=$_.Group[0].original_split; assigned_split=$_.Group[0].split; count=$_.Count }
})
$originalTestMovedToTrain = @($auditRows | Where-Object { $_.original_split -eq 'test' -and $_.split -eq 'train' })
if ($originalTestMovedToTrain.Count) {
    $issues.Add((New-Issue medium source_test_reassigned_to_train $originalTestMovedToTrain.Count "Safe for the new V2 split itself, but contaminates evaluation against either source's original test split."))
}

$samplesCsv = Join-Path $outputAbsolute 'samples.csv'
$exactCsv = Join-Path $outputAbsolute 'exact_duplicates.csv'
$allNearCsv = Join-Path $outputAbsolute 'all_near_duplicates.csv'
$nearCsv = Join-Path $outputAbsolute 'cross_split_near_duplicates.csv'
$componentsCsv = Join-Path $outputAbsolute 'duplicate_components.csv'
$issuesCsv = Join-Path $outputAbsolute 'issues.csv'
$auditRows | Select-Object sample_id,tumor_type,source_dataset,original_split,split,group_id,image_path,mask_path,image_pixel_sha256,mask_pixel_sha256,image_distinct_values,image_min,image_max,image_mean,image_stddev,mask_distinct_values,mask_binary,foreground_pixels,foreground_fraction,mask_min_x,mask_min_y,mask_max_x,mask_max_y,actual_image_sha256,actual_mask_sha256 | Export-Csv -LiteralPath $samplesCsv -NoTypeInformation -Encoding UTF8
$exactRows | Export-Csv -LiteralPath $exactCsv -NoTypeInformation -Encoding UTF8
$nearRows | Export-Csv -LiteralPath $allNearCsv -NoTypeInformation -Encoding UTF8
$nearRows | Where-Object { $_.first_split -ne $_.second_split } | Export-Csv -LiteralPath $nearCsv -NoTypeInformation -Encoding UTF8
$duplicateComponentRows | Export-Csv -LiteralPath $componentsCsv -NoTypeInformation -Encoding UTF8
$issues | Export-Csv -LiteralPath $issuesCsv -NoTypeInformation -Encoding UTF8

$summary = [ordered]@{
    schema_version=1; created_at_utc=[DateTime]::UtcNow.ToString('o'); dataset=Relative-Path $datasetAbsolute
    scope='File/sample-level audit; patient-level leakage intentionally excluded.'
    thresholds=[ordered]@{ maximum_dhash_distance=$MaximumHammingDistance; minimum_pixel_correlation=$MinimumCorrelation; strong_dhash_distance=8; strong_pixel_correlation=0.98; likely_duplicate_mask_iou=0.90; orientations=8 }
    counts=[ordered]@{ samples=$auditRows.Count; filesystem_files=$filesystemFiles.Count; train=@($auditRows | Where-Object split -eq train).Count; val=@($auditRows | Where-Object split -eq val).Count; test=@($auditRows | Where-Object split -eq test).Count }
    integrity=[ordered]@{ clean_only_rows=$cleanOnly.Count; split_only_rows=$splitOnly.Count; missing_pairs=$missingPairs.Count; hash_mismatches=$badHash.Count; bad_dimensions=$badDimensions.Count; unexpected_bit_depth=$badBpp.Count; invalid_masks=$badMasks.Count; low_information_images=$lowInformation.Count; unreferenced_files=$unreferenced.Count; declared_group_leaks=$groupLeaks.Count }
    duplicates=[ordered]@{ exact_image_groups=$imageDuplicateGroups.Count; exact_mask_groups=$maskDuplicateGroups.Count; exact_pair_groups=$pairDuplicateGroups.Count; cross_split_exact_image_groups=$crossSplitImageGroups.Count; cross_split_exact_mask_groups=$crossSplitMaskGroups.Count; cross_split_exact_pair_groups=$crossSplitPairGroups.Count; all_relaxed_near_pairs=$nearRows.Count; all_likely_duplicate_pairs=$allLikelyNear.Count; within_split_likely_duplicate_pairs=$withinSplitLikely.Count; cross_split_strong_near_pairs=$strongNear.Count; cross_split_likely_duplicate_pairs=$likelyNear.Count; cross_class_likely_duplicate_pairs=$crossClassLikely.Count; duplicate_components=$duplicateComponents.Count; duplicate_affected_samples=$duplicateIds.Count; cross_split_duplicate_components=$leakingDuplicateComponents.Count; cross_split_affected_samples=@($leakingDuplicateComponents.Group.sample_id | Sort-Object -Unique).Count }
    benchmark_provenance=[ordered]@{ original_test_reassigned_to_train=$originalTestMovedToTrain.Count; transition_counts=$transition }
    issues=@($issues)
    artifacts=[ordered]@{ samples=Relative-Path $samplesCsv; exact_duplicates=Relative-Path $exactCsv; all_near_duplicates=Relative-Path $allNearCsv; cross_split_near_duplicates=Relative-Path $nearCsv; duplicate_components=Relative-Path $componentsCsv; issues=Relative-Path $issuesCsv }
}
$summaryPath = Join-Path $outputAbsolute 'summary.json'
[IO.File]::WriteAllText($summaryPath, (($summary | ConvertTo-Json -Depth 10) + [Environment]::NewLine), (New-Object Text.UTF8Encoding($false)))
Write-Output "Audit complete: $summaryPath"
Write-Output "Samples=$($auditRows.Count), exact image groups=$($imageDuplicateGroups.Count), cross-split strong near pairs=$($strongNear.Count), likely duplicates=$($likelyNear.Count)"
