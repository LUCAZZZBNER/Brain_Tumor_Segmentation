[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$DatasetRoot = "DATASET/kaggle_3m",
    [string]$OutputDirectory = "reports/kaggle_3m_audit",
    [int]$MaximumDHashDistance = 12,
    [double]$CandidateMinimumCorrelation = 0.95,
    [double]$DuplicateMinimumCorrelation = 0.998,
    [double]$DuplicateMaximumMae = 3.0
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Value))
}
function Relative-ProjectPath([string]$Absolute) {
    $rootUri = [Uri](($ProjectRoot.TrimEnd('\') + '\'))
    return [Uri]::UnescapeDataString($rootUri.MakeRelativeUri([Uri]$Absolute).ToString())
}
function Write-Utf8Json([object]$Value, [string]$PathValue) {
    [IO.File]::WriteAllText($PathValue, (($Value | ConvertTo-Json -Depth 12) + [Environment]::NewLine), (New-Object Text.UTF8Encoding($false)))
}

$datasetAbsolute = Resolve-ProjectPath $DatasetRoot
$outputAbsolute = Resolve-ProjectPath $OutputDirectory
New-Item -ItemType Directory -Path $outputAbsolute -Force | Out-Null
Add-Type -AssemblyName System.Drawing
Add-Type -Path (Join-Path $PSScriptRoot 'V2ImageTools.cs') -ReferencedAssemblies System.Drawing
Add-Type -Path (Join-Path $PSScriptRoot 'SegmentationV2AuditTools.cs') -ReferencedAssemblies System.Drawing

Write-Output '[1/5] Discovering patient folders and image/mask pairs'
$patientDirectories = @(Get-ChildItem -LiteralPath $datasetAbsolute -Directory | Sort-Object Name)
$allTiffs = @(Get-ChildItem -LiteralPath $datasetAbsolute -Recurse -File -Filter *.tif)
$images = @($allTiffs | Where-Object { $_.BaseName -notlike '*_mask' })
$masks = @($allTiffs | Where-Object { $_.BaseName -like '*_mask' })
$maskByPath = @{}
foreach ($mask in $masks) { $maskByPath[$mask.FullName] = $mask }
$records = [Collections.Generic.List[object]]::new()
$missingMasks = [Collections.Generic.List[string]]::new()
foreach ($image in $images | Sort-Object FullName) {
    $maskPath = Join-Path $image.DirectoryName ($image.BaseName + '_mask' + $image.Extension)
    if (-not (Test-Path -LiteralPath $maskPath -PathType Leaf)) { $missingMasks.Add($image.FullName); continue }
    $sliceText = $image.BaseName.Substring($image.Directory.Name.Length + 1)
    $sliceNumber = 0
    [void][int]::TryParse($sliceText, [ref]$sliceNumber)
    $records.Add([pscustomobject]@{
        Index=$records.Count; PatientId=$image.Directory.Name; SliceNumber=$sliceNumber
        ImagePath=$image.FullName; MaskPath=$maskPath
        ImageRelative=Relative-ProjectPath $image.FullName; MaskRelative=Relative-ProjectPath $maskPath
    })
}
$expectedMaskPaths = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($record in $records) { [void]$expectedMaskPaths.Add($record.MaskPath) }
$orphanMasks = @($masks | Where-Object { -not $expectedMaskPaths.Contains($_.FullName) })
if ($missingMasks.Count -or $orphanMasks.Count) { Write-Warning "Missing masks=$($missingMasks.Count), orphan masks=$($orphanMasks.Count)" }

Write-Output '[2/5] Decoding every RGB image and binary mask'
$inspectionLines = [SegmentationV2AuditTools]::InspectRgbMaskPairs(
    [string[]]@($records.ImagePath), [string[]]@($records.MaskPath))
foreach ($line in $inspectionLines) {
    $p = $line -split '\|'; $row = $records[[int]$p[0]]
    $row | Add-Member NoteProperty ImageWidth ([int]$p[1]); $row | Add-Member NoteProperty ImageHeight ([int]$p[2]); $row | Add-Member NoteProperty ImageBpp ([int]$p[3])
    $row | Add-Member NoteProperty MaskWidth ([int]$p[4]); $row | Add-Member NoteProperty MaskHeight ([int]$p[5]); $row | Add-Member NoteProperty MaskBpp ([int]$p[6])
    $row | Add-Member NoteProperty ImagePixelSha256 $p[7]; $row | Add-Member NoteProperty MaskPixelSha256 $p[8]
    $row | Add-Member NoteProperty MaskDistinctValues ([int]$p[9]); $row | Add-Member NoteProperty MaskBinary ([int]$p[10] -eq 1)
    $row | Add-Member NoteProperty ForegroundPixels ([long]$p[11]); $row | Add-Member NoteProperty ForegroundFraction ([long]$p[11] / ([int]$p[4]*[int]$p[5]))
    $row | Add-Member NoteProperty RedGreenEqual ([int]$p[12] -eq 1); $row | Add-Member NoteProperty RedBlueEqual ([int]$p[13] -eq 1); $row | Add-Member NoteProperty GreenBlueEqual ([int]$p[14] -eq 1)
    $row | Add-Member NoteProperty RedMin ([int]$p[15]); $row | Add-Member NoteProperty RedMax ([int]$p[16]); $row | Add-Member NoteProperty RedStd ([double]::Parse($p[17],[Globalization.CultureInfo]::InvariantCulture))
    $row | Add-Member NoteProperty GreenMin ([int]$p[18]); $row | Add-Member NoteProperty GreenMax ([int]$p[19]); $row | Add-Member NoteProperty GreenStd ([double]::Parse($p[20],[Globalization.CultureInfo]::InvariantCulture))
    $row | Add-Member NoteProperty BlueMin ([int]$p[21]); $row | Add-Member NoteProperty BlueMax ([int]$p[22]); $row | Add-Member NoteProperty BlueStd ([double]::Parse($p[23],[Globalization.CultureInfo]::InvariantCulture))
    $row | Add-Member NoteProperty ImageFileSha256 ([V2ImageTools]::Sha256File($row.ImagePath)); $row | Add-Member NoteProperty MaskFileSha256 ([V2ImageTools]::Sha256File($row.MaskPath))
}

$badDimensions = @($records | Where-Object { $_.ImageWidth -ne 256 -or $_.ImageHeight -ne 256 -or $_.MaskWidth -ne 256 -or $_.MaskHeight -ne 256 -or $_.ImageWidth -ne $_.MaskWidth -or $_.ImageHeight -ne $_.MaskHeight })
$invalidMasks = @($records | Where-Object { -not $_.MaskBinary })
$emptyMasks = @($records | Where-Object ForegroundPixels -eq 0)
$fullMasks = @($records | Where-Object ForegroundPixels -eq 65536)
$lowInformationImages = @($records | Where-Object { $_.RedStd -lt 3 -and $_.GreenStd -lt 3 -and $_.BlueStd -lt 3 })

Write-Output '[3/5] Checking exact decoded-content duplicates'
$imageDuplicateGroups = @($records | Group-Object ImagePixelSha256 | Where-Object Count -gt 1)
$crossPatientImageGroups = @($imageDuplicateGroups | Where-Object { @($_.Group.PatientId | Sort-Object -Unique).Count -gt 1 })
$nonemptyMaskDuplicateGroups = @($records | Where-Object ForegroundPixels -gt 0 | Group-Object MaskPixelSha256 | Where-Object Count -gt 1)
$crossPatientNonemptyMaskGroups = @($nonemptyMaskDuplicateGroups | Where-Object { @($_.Group.PatientId | Sort-Object -Unique).Count -gt 1 })

Write-Output '[4/5] Searching orientation-aware near-duplicates'
$uniqueLabels = [string[]]@(0..($records.Count-1) | ForEach-Object { "sample_$_" })
$candidateLines = [SegmentationV2AuditTools]::FindCrossSplitNearDuplicates(
    [string[]]@($records.ImagePath), [string[]]@($records.MaskPath), $uniqueLabels,
    $MaximumDHashDistance, $CandidateMinimumCorrelation)
$nearRows = @(
    foreach ($line in $candidateLines) {
        $p=$line -split '\|'; $first=$records[[int]$p[0]]; $second=$records[[int]$p[1]]
        [pscustomobject]@{
            FirstIndex=[int]$p[0];SecondIndex=[int]$p[1]
            FirstPatient=$first.PatientId;FirstSlice=$first.SliceNumber;FirstImage=$first.ImageRelative
            SecondPatient=$second.PatientId;SecondSlice=$second.SliceNumber;SecondImage=$second.ImageRelative
            SamePatient=($first.PatientId -eq $second.PatientId);DHashDistance=[int]$p[2]
            CandidateCorrelation=[double]::Parse($p[3],[Globalization.CultureInfo]::InvariantCulture)
            CandidateMaskIoU=[double]::Parse($p[4],[Globalization.CultureInfo]::InvariantCulture);Transform=$p[5]
        }
    }
)
$verificationLines = [SegmentationV2AuditTools]::VerifyTransformedPairs(
    [string[]]@($nearRows | ForEach-Object {$records[$_.FirstIndex].ImagePath}),
    [string[]]@($nearRows | ForEach-Object {$records[$_.FirstIndex].MaskPath}),
    [string[]]@($nearRows | ForEach-Object {$records[$_.SecondIndex].ImagePath}),
    [string[]]@($nearRows | ForEach-Object {$records[$_.SecondIndex].MaskPath}),
    [string[]]@($nearRows.Transform))
foreach ($line in $verificationLines) {
    $p=$line -split '\|';$row=$nearRows[[int]$p[0]]
    $row|Add-Member NoteProperty FullCorrelation ([double]::Parse($p[1],[Globalization.CultureInfo]::InvariantCulture))
    $row|Add-Member NoteProperty ImageMae ([double]::Parse($p[2],[Globalization.CultureInfo]::InvariantCulture))
    $row|Add-Member NoteProperty ImageRmse ([double]::Parse($p[3],[Globalization.CultureInfo]::InvariantCulture))
    $row|Add-Member NoteProperty FullMaskIoU ([double]::Parse($p[6],[Globalization.CultureInfo]::InvariantCulture))
    $row|Add-Member NoteProperty StrongDuplicate ($row.FullCorrelation -ge $DuplicateMinimumCorrelation -and $row.ImageMae -le $DuplicateMaximumMae)
}
$strongPairs = @($nearRows | Where-Object StrongDuplicate)
$crossPatientStrongPairs = @($strongPairs | Where-Object { -not $_.SamePatient })

Write-Output '[5/5] Checking metadata and writing audit artifacts'
$metadataPath = Join-Path $datasetAbsolute 'data.csv'
$metadataRows = @(Import-Csv -LiteralPath $metadataPath)
$metadataPatients = New-Object 'Collections.Generic.HashSet[string]'
foreach ($row in $metadataRows) { [void]$metadataPatients.Add($row.Patient) }
$folderPatients = @($patientDirectories | ForEach-Object { $_.Name -replace '_\d{8}$','' } | Sort-Object -Unique)
$foldersMissingMetadata = @($folderPatients | Where-Object { -not $metadataPatients.Contains($_) })
$metadataMissingFolders = @($metadataRows.Patient | Where-Object { $folderPatients -notcontains $_ })

$samplesCsv=Join-Path $outputAbsolute 'samples.csv';$nearCsv=Join-Path $outputAbsolute 'near_duplicates.csv';$crossCsv=Join-Path $outputAbsolute 'cross_patient_strong_duplicates.csv'
$records | Select-Object PatientId,SliceNumber,ImageRelative,MaskRelative,ImageWidth,ImageHeight,ImageBpp,MaskWidth,MaskHeight,MaskBpp,ImagePixelSha256,MaskPixelSha256,ForegroundPixels,ForegroundFraction,MaskDistinctValues,MaskBinary,RedGreenEqual,RedBlueEqual,GreenBlueEqual,RedStd,GreenStd,BlueStd,ImageFileSha256,MaskFileSha256 | Export-Csv -LiteralPath $samplesCsv -NoTypeInformation -Encoding UTF8
$nearRows | Export-Csv -LiteralPath $nearCsv -NoTypeInformation -Encoding UTF8
$crossPatientStrongPairs | Export-Csv -LiteralPath $crossCsv -NoTypeInformation -Encoding UTF8
$summary=[ordered]@{
    schema_version=1;created_at_utc=[DateTime]::UtcNow.ToString('o');dataset=Relative-ProjectPath $datasetAbsolute
    scope='Raw-dataset audit. No train/validation/test manifest exists; within-patient similarity is not treated as leakage.'
    counts=[ordered]@{patient_folders=$patientDirectories.Count;metadata_patients=$metadataRows.Count;pairs=$records.Count;tiff_files=$allTiffs.Count;missing_masks=$missingMasks.Count;orphan_masks=$orphanMasks.Count}
    integrity=[ordered]@{bad_dimensions=$badDimensions.Count;invalid_masks=$invalidMasks.Count;empty_masks=$emptyMasks.Count;full_masks=$fullMasks.Count;low_information_images=$lowInformationImages.Count;folders_missing_metadata=$foldersMissingMetadata.Count;metadata_missing_folders=$metadataMissingFolders.Count}
    channels=[ordered]@{red_equals_green_images=@($records|Where-Object RedGreenEqual).Count;green_equals_blue_images=@($records|Where-Object GreenBlueEqual).Count;all_channels_equal_images=@($records|Where-Object {$_.RedGreenEqual-and$_.GreenBlueEqual}).Count;patients_with_red_green_equality=@($records|Where-Object RedGreenEqual|Select-Object -ExpandProperty PatientId -Unique).Count;patients_with_green_blue_equality=@($records|Where-Object GreenBlueEqual|Select-Object -ExpandProperty PatientId -Unique).Count}
    exact_duplicates=[ordered]@{image_groups=$imageDuplicateGroups.Count;cross_patient_image_groups=$crossPatientImageGroups.Count;empty_mask_repetitions=$emptyMasks.Count;nonempty_mask_groups=$nonemptyMaskDuplicateGroups.Count;cross_patient_nonempty_mask_groups=$crossPatientNonemptyMaskGroups.Count}
    near_duplicates=[ordered]@{relaxed_candidates=$nearRows.Count;strong_pairs=$strongPairs.Count;same_patient_strong_pairs=@($strongPairs|Where-Object SamePatient).Count;cross_patient_strong_pairs=$crossPatientStrongPairs.Count;thresholds=[ordered]@{minimum_full_correlation=$DuplicateMinimumCorrelation;maximum_image_mae=$DuplicateMaximumMae;orientations=8}}
    mask_distribution=[ordered]@{positive_masks=@($records|Where-Object ForegroundPixels -gt 0).Count;empty_masks=$emptyMasks.Count;empty_fraction=$emptyMasks.Count/[double]$records.Count;mean_foreground_fraction=(($records|Measure-Object ForegroundFraction -Average).Average);empty_prediction_macro_iou_under_current_metric=$emptyMasks.Count/[double]$records.Count}
    split_risk='The raw dataset has no split. Slice-level random splitting would create patient-level leakage and can also distribute any duplicate groups across splits.'
    artifacts=[ordered]@{samples=Relative-ProjectPath $samplesCsv;near_duplicates=Relative-ProjectPath $nearCsv;cross_patient_duplicates=Relative-ProjectPath $crossCsv}
}
Write-Utf8Json $summary (Join-Path $outputAbsolute 'summary.json')
Write-Output "Audit complete: pairs=$($records.Count), empty masks=$($emptyMasks.Count), cross-patient strong duplicates=$($crossPatientStrongPairs.Count)"
