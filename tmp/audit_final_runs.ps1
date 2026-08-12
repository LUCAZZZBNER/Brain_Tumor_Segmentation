$expectedManifest='6aee075d9ea8b4a41cdefc372e381a55ec3a099acfaf16002dbde2cd7d5051e7'
$models=[ordered]@{
 'E0'='kaggle_3m_multimodal_only_e0_flair_unet'
 'E1-A'='kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation'
 'E2-B'='kaggle_3m_multimodal_only_e2_flair_unet_augmentation'
 'M0-AB'='kaggle_3m_multimodal_only_m0_rgb_unet'
 'M4-NP'='kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet'
 'M4-P−A'='kaggle_3m_multimodal_only_m4_p_minus_a_flair_resnet34_unet'
 'M4-P−B'='kaggle_3m_multimodal_only_m4_p_minus_b_rgb_resnet34_unet_no_augmentation'
 'M4-P'='kaggle_3m_multimodal_only_m4_rgb_resnet34_unet'
}
$rows=@()
foreach($name in $models.Keys){
  $vals=@(); $dice=@(); $miou=@(); $prec=@(); $rec=@(); $fp=@()
  foreach($seed in 42,123,2026){
    $p="runs/$($models[$name])_seed$seed/test_metrics.json"
    if(-not(Test-Path $p)){throw "Missing $p"}
    $j=Get-Content -Raw $p|ConvertFrom-Json
    if($j.manifest_sha256 -ne $expectedManifest){throw "Manifest mismatch $p"}
    if($j.metrics.num_samples -ne 525 -or $j.metrics.num_positive_images -ne 173 -or $j.metrics.num_empty_images -ne 352){throw "Count mismatch $p"}
    $vals += [double]$j.metrics.positive_macro_iou
    $dice += [double]$j.metrics.positive_macro_dice
    $miou += [double]$j.metrics.micro_iou
    $prec += [double]$j.metrics.micro_precision
    $rec += [double]$j.metrics.micro_recall
    $fp += 100*[double]$j.metrics.empty_slice_false_positive_rate
  }
  function Mean($x){($x|Measure-Object -Average).Average}
  function SD($x){$m=Mean $x; [Math]::Sqrt((($x|%{($_-$m)*($_-$m)}|Measure-Object -Sum).Sum)/($x.Count-1))}
  $rows += [pscustomobject]@{Model=$name;PositiveIoU=('{0:F4} ± {1:F4}' -f (Mean $vals),(SD $vals));PositiveDice=('{0:F4} ± {1:F4}' -f (Mean $dice),(SD $dice));MicroIoU=('{0:F4} ± {1:F4}' -f (Mean $miou),(SD $miou));Precision=('{0:F4} ± {1:F4}' -f (Mean $prec),(SD $prec));Recall=('{0:F4} ± {1:F4}' -f (Mean $rec),(SD $rec));EmptyFP=('{0:F2}% ± {1:F2}%' -f (Mean $fp),(SD $fp))}
}
$rows
"AUDIT_OK runs=24 manifest=$expectedManifest samples=525 positive=173 empty=352"
