param(
  [Parameter(Mandatory=$true)][string]$Source,
  [Parameter(Mandatory=$true)][string]$Output,
  [Parameter(Mandatory=$true)][string]$ArchitectureImage
)
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression
$src=(Resolve-Path -LiteralPath $Source).Path
$img=(Resolve-Path -LiteralPath $ArchitectureImage).Path
$out=[IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
Copy-Item -LiteralPath $src -Destination $out -Force
$zip=[IO.Compression.ZipFile]::Open($out,[IO.Compression.ZipArchiveMode]::Update)
try {
  function ReadPart($name){$e=$zip.GetEntry($name);if(-not$e){throw "Missing $name"};$r=[IO.StreamReader]::new($e.Open());try{$r.ReadToEnd()}finally{$r.Dispose()}}
  function WritePart($name,[xml]$doc){$old=$zip.GetEntry($name);if($old){$old.Delete()};$e=$zip.CreateEntry($name,[IO.Compression.CompressionLevel]::Optimal);$s=$e.Open();$set=[Xml.XmlWriterSettings]::new();$set.Encoding=[Text.UTF8Encoding]::new($false);$set.Indent=$false;$w=[Xml.XmlWriter]::Create($s,$set);try{$doc.Save($w)}finally{$w.Dispose();$s.Dispose()}}
  $x=[Xml.XmlDocument]::new();$x.PreserveWhitespace=$true;$x.LoadXml((ReadPart 'word/document.xml'))
  $wuri='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
  $ruri='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
  $auri='http://schemas.openxmlformats.org/drawingml/2006/main'
  $wpuri='http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
  $picuri='http://schemas.openxmlformats.org/drawingml/2006/picture'
  $n=[Xml.XmlNamespaceManager]::new($x.NameTable);$n.AddNamespace('w',$wuri);$n.AddNamespace('r',$ruri);$n.AddNamespace('a',$auri);$n.AddNamespace('wp',$wpuri);$n.AddNamespace('pic',$picuri)
  function PText($p){(($p.SelectNodes('.//w:t',$n)|% InnerText)-join '')}
  function FindP($text){$m=@($x.SelectNodes('//w:body//w:p',$n)|?{(PText $_)-eq$text});if($m.Count-ne1){throw "Paragraph [$text] count=$($m.Count)"};$m[0]}
  function SetP($p,$text){foreach($c in @($p.ChildNodes)){if(-not($c.LocalName-eq'pPr'-and$c.NamespaceURI-eq$wuri)){[void]$p.RemoveChild($c)}};$r=$x.CreateElement('w','r',$wuri);$t=$x.CreateElement('w','t',$wuri);$t.InnerText=$text;[void]$r.AppendChild($t);[void]$p.AppendChild($r)}
  function NewPFrom($template,$text){$p=$template.CloneNode($true);SetP $p $text;$p}
  function ReplaceBody($headingText,$nextHeadingText,[string[]]$texts){$h=FindP $headingText;$next=FindP $nextHeadingText;$nodes=@();$c=$h.NextSibling;while($c-and$c-ne$next){$nn=$c.NextSibling;if($c.LocalName-eq'p'){$nodes+=$c};$c=$nn};if($nodes.Count-lt1){throw "No body after $headingText"};$tpl=$nodes[0].CloneNode($true);foreach($q in $nodes){[void]$q.ParentNode.RemoveChild($q)};foreach($t in $texts){$p=NewPFrom $tpl $t;[void]$next.ParentNode.InsertBefore($p,$next)}}
  function InsertBeforeTableByFirstCell($firstCell,$caption,$template){$tbl=@($x.SelectNodes('//w:body/w:tbl',$n)|?{(($_.SelectNodes('./w:tr[1]/w:tc[1]//w:t',$n)|% InnerText)-join '')-eq$firstCell});if($tbl.Count-ne1){throw "Table [$firstCell] count=$($tbl.Count)"};$p=NewPFrom $template $caption;[void]$tbl[0].ParentNode.InsertBefore($p,$tbl[0])}

  # 全文连续图号与表号。
  $repl=[ordered]@{
    '图 3-1'='图1'; '图3-1'='图1'; '图 4-1'='图3'; '图4-1'='图3';
    '图 4-2'='图4'; '图4-2'='图4'; '图 4-3'='图5'; '图4-3'='图5';
    '图 4-4'='图6'; '图4-4'='图6'; '表 4-1'='表2'; '表4-1'='表2'
  }
  foreach($p in $x.SelectNodes('//w:body//w:p',$n)){$txt=PText $p;if($txt){$new=$txt;foreach($k in $repl.Keys){$new=$new.Replace($k,$repl[$k])};if($new-ne$txt){SetP $p $new}}}

  $tableCaption=FindP '表2 实验环境配置'
  InsertBeforeTableByFirstCell '项目' '表1 数据集清洁与患者级划分统计' $tableCaption
  InsertBeforeTableByFirstCell '配置' '表3 八种模型配置与组件对照' $tableCaption
  InsertBeforeTableByFirstCell '模型' '表4 八种模型测试集分割性能（均值±样本标准差）' $tableCaption
  InsertBeforeTableByFirstCell '组件' '表5 组件消融的严格配对结果' $tableCaption

  # 数据来源、许可证及通道说明。
  $old='当前主实验使用仓库中的 DATASET/kaggle_3m，具体来源名称和许可证信息应在正式稿中依据原始下载页面核对。'
  $new='本文使用 Kaggle 的 Brain MRI segmentation（LGG Segmentation Dataset）公开数据集[16]。数据页说明，图像来源于 The Cancer Imaging Archive（TCIA），对应 The Cancer Genome Atlas（TCGA）低级别胶质瘤集合中的 110 名患者，并提供人工 FLAIR 异常区域掩膜；Kaggle 页面标注的许可证为 CC BY-NC-SA 4.0。本文未修改原始文件，仅通过患者级清洁清单排除 6 名灰度等价患者，最终纳入 104 名患者和 3,629 张切片。'
  SetP (FindP $old) $new

  # 将模型结构图嵌入3.5末尾，图注置于图下。
  $rels=[Xml.XmlDocument]::new();$rels.PreserveWhitespace=$true;$rels.LoadXml((ReadPart 'word/_rels/document.xml.rels'))
  $relUri='http://schemas.openxmlformats.org/package/2006/relationships'
  $max=0;foreach($a in $rels.DocumentElement.ChildNodes){if($a.Id-match'^rId(\d+)$'){$max=[Math]::Max($max,[int]$Matches[1])}}
  $rid="rId$($max+1)";$rel=$rels.CreateElement('Relationship',$relUri);$rel.SetAttribute('Id',$rid);$rel.SetAttribute('Type','http://schemas.openxmlformats.org/officeDocument/2006/relationships/image');$rel.SetAttribute('Target','media/m4_p_resnet34_unet_architecture.png');[void]$rels.DocumentElement.AppendChild($rel)
  $oldMedia=$zip.GetEntry('word/media/m4_p_resnet34_unet_architecture.png');if($oldMedia){$oldMedia.Delete()};$me=$zip.CreateEntry('word/media/m4_p_resnet34_unet_architecture.png',[IO.Compression.CompressionLevel]::Optimal);$ms=$me.Open();$fs=[IO.File]::OpenRead($img);try{$fs.CopyTo($ms)}finally{$fs.Dispose();$ms.Dispose()}
  $imageTemplate=@($x.SelectNodes('//w:body//w:p[.//w:drawing]',$n))[0].CloneNode($true)
  $blip=$imageTemplate.SelectSingleNode('.//a:blip',$n);$blip.SetAttribute('embed',$ruri,$rid)
  $docPr=$imageTemplate.SelectSingleNode('.//wp:docPr',$n);if($docPr){$ids=@($x.SelectNodes('//wp:docPr',$n)|%{[int]$_.GetAttribute('id')});$docPr.SetAttribute('id',([string](($ids|Measure-Object -Maximum).Maximum+1)));$docPr.SetAttribute('name','M4-P ResNet34-U-Net architecture')}
  foreach($extent in @($imageTemplate.SelectNodes('.//wp:extent | .//a:xfrm/a:ext',$n))){$extent.SetAttribute('cx','5943600');$extent.SetAttribute('cy','2904480')}
  $bodyPara=FindP '两类模型均输出单通道病灶概率图，可接收单通道 FLAIR 或三通道输入。为区分输入、增强、编码器结构和预训练的影响，具体模型配置与对照关系统一放在第 4.3 节说明。'
  SetP $bodyPara '两类模型均输出单通道病灶概率图，可接收单通道 FLAIR 或三通道输入。M4-P 的 ResNet34-U-Net 结构如图2所示：编码器依次输出多尺度特征，解码器通过转置卷积恢复空间分辨率，并与对应尺度的编码特征进行跳跃连接。为区分输入、增强、编码器结构和预训练的影响，具体配置与对照关系统一放在第 4.3 节说明。'
  $after=$bodyPara.NextSibling;[void]$bodyPara.ParentNode.InsertBefore($imageTemplate,$after)
  $figCapTemplate=FindP '图1 本文研究的总体流程'
  $cap=NewPFrom $figCapTemplate '图2 M4-P 的 ResNet34-U-Net 模型结构'
  [void]$bodyPara.ParentNode.InsertBefore($cap,$after)

  # 将可复核的一致性审计写入多随机种子实验。
  $multi=FindP '所有八种配置使用相同清单，训练随机种子为 42、123 和 2026。结果报告均值和样本标准差，不从三个种子中挑选最高一次作为最终结果。'
  $audit=NewPFrom $multi '成稿前重新核对了 8 种配置、3 个训练随机种子的 24 份 test_metrics.json。所有记录的 manifest_sha256 均为 6aee075d9ea8b4a41cdefc372e381a55ec3a099acfaf16002dbde2cd7d5051e7，与 splits/kaggle_3m_multimodal_only_seed42.meta.json 一致；每次测试均包含 525 张切片，其中阳性切片 173 张、空切片 352 张。由原始运行记录重新计算所得表4各项均值与样本标准差均与正文一致。'
  [void]$multi.ParentNode.InsertAfter($audit,$multi)

  # 第六部分适度扩充，保持简洁。
  ReplaceBody '6.1 结论（当前证据版）' '6.2 未来工作' @(
    '本文围绕二维脑肿瘤 MRI 病灶分割建立了从数据核验、患者级划分、模型训练到多指标评估的可复现实验流程。基于 104 名患者和 3,629 张切片，本文在同一清洁清单和训练协议下比较了八种模型配置，并通过 42、123 和 2026 三个随机种子的严格配对实验考察三通道输入、轻量增强、ResNet34 编码器和 ImageNet 预训练的作用。',
    '实验结果显示，完整方案 M4-P 的测试集 Positive Macro IoU 和 Positive Dice 分别为 0.7664±0.0086 和 0.8425±0.0101；相对 E0 的 Positive Macro IoU 配对提升为 0.1129±0.0249，三个随机种子的提升方向一致。ResNet34 编码器配置表现出相对稳定的正向贡献，而三通道输入、轻量增强和 ImageNet 预训练的独立效果仍受随机种子和患者差异影响。误差分析同时表明，小病灶漏检和空切片误报仍是主要问题。因此，本文结论仅适用于当前公开数据、患者划分和实验协议，不能直接视为临床泛化或诊断有效性证据。'
  )
  ReplaceBody '6.2 未来工作' '参考文献' @(
    '后续首先应冻结当前模型、阈值和评价流程，在独立医院或不同设备来源的数据上开展患者级外部验证，并增加患者数量或患者级交叉验证，以更可靠地估计病例间差异。',
    '在方法方面，可进一步考察 2.5D 或 3D 上下文、小病灶重采样、边界敏感损失、阈值校准与不确定性估计，同时将完整多模态输入与缺失模态处理分开评价。',
    '在应用评价方面，应邀请影像科专家盲法复核分割边界、完全漏检和严重误报，并结合人工修订时间评估模型的实际辅助价值。在完成上述验证前，模型仍定位为研究性工具。'
  )

  # 增补数据集网页参考文献，并删除已完成的待办清单。
  $ref15=FindP '[15] ZHANG Y, HAN Y, ZHANG J. MAU-Net: Mixed attention U-Net for MRI brain tumor segmentation[J]. Mathematical Biosciences and Engineering, 2023, 20(12): 20510-20527. DOI: 10.3934/mbe.2023907. PMID: 38124563.'
  $ref16=NewPFrom $ref15 '[16] BUDA M. Brain MRI segmentation: LGG Segmentation Dataset[DB/OL]. Kaggle. https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation (accessed 2026-08-12). License: CC BY-NC-SA 4.0.'
  [void]$ref15.ParentNode.InsertAfter($ref16,$ref15)
  $todo=FindP '需要在成稿前补齐的材料清单';$c=$todo;while($c){$next=$c.NextSibling;[void]$c.ParentNode.RemoveChild($c);$c=$next}

  WritePart 'word/document.xml' $x;WritePart 'word/_rels/document.xml.rels' $rels
}finally{$zip.Dispose()}
$out
