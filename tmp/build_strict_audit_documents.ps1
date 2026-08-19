param(
  [string]$Source = 'reports\论文中文草稿框架_第三至六部分完善终稿_会议论文结构再精简版.docx',
  [string]$Revision = 'reports\论文中文草稿框架_第三至六部分完善终稿_会议论文结构再精简版_严格证据修订稿.docx',
  [string]$Report = 'reports\论文实验数据与参考文献严格审计报告.docx',
  [string]$AuditJson = 'tmp\strict_evidence_audit.json'
)

$ErrorActionPreference = 'Stop'
trap {
  Write-Error ("STRICT AUDIT BUILD FAILED line {0}: {1}" -f $_.InvocationInfo.ScriptLineNumber, $_.Exception.Message)
  break
}
$root = (Resolve-Path '.').Path
$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$auditPath = (Resolve-Path -LiteralPath $AuditJson).Path
$revisionPath = [IO.Path]::GetFullPath((Join-Path $root $Revision))
$reportPath = [IO.Path]::GetFullPath((Join-Path $root $Report))
$audit = Get-Content -Raw -Encoding UTF8 $auditPath | ConvertFrom-Json

function Get-FullDocumentText($doc) {
  return ($doc.Content.Text -replace "`r", "`n")
}

function Find-ExactParagraph($doc, [string]$text) {
  foreach ($p in $doc.Paragraphs) {
    $value = ($p.Range.Text -replace "[`r`a]", '').Trim()
    if ($value -eq $text) { return $p }
  }
  throw "Paragraph not found: $text"
}

function Replace-Paragraph($doc, [string]$oldText, [string]$newText, [string]$comment = '') {
  $p = Find-ExactParagraph $doc $oldText
  $range = $p.Range.Duplicate
  $range.End = $range.End - 1
  $range.Text = $newText
  if ($comment) {
    $anchor = $doc.Range($range.Start, [Math]::Min($range.Start + $newText.Length, $range.End))
    [void]$doc.Comments.Add($anchor, $comment)
  }
}

function Add-Comment-ToParagraph($doc, [string]$text, [string]$comment) {
  $p = Find-ExactParagraph $doc $text
  $r = $p.Range.Duplicate
  $r.End = $r.End - 1
  [void]$doc.Comments.Add($r, $comment)
}

function Set-CellText($cell, [string]$text) {
  $cell.Range.Text = $text
}

function Apply-TableStyle($table, $word) {
  $table.AllowAutoFit = $true
  $table.Rows.AllowBreakAcrossPages = 0
  $table.Rows.Item(1).HeadingFormat = -1
  $table.Rows.Item(1).Range.Bold = 1
  $table.Rows.Item(1).Range.Shading.BackgroundPatternColor = 15132390
  $table.Range.Font.Name = '宋体'
  $table.Range.Font.NameFarEast = '宋体'
  $table.Range.Font.Size = 9
  $table.Range.ParagraphFormat.SpaceAfter = 0
  $table.Range.ParagraphFormat.SpaceBefore = 0
  $table.Range.ParagraphFormat.LineSpacingRule = 0
  $table.Rows.Alignment = 1
  $table.Borders.Enable = 1
}

function Add-Heading($doc, [string]$text, [int]$level) {
  $start = $doc.Content.End - 1
  $insert = $doc.Range($start, $start)
  $insert.InsertAfter($text + "`r")
  $p = $doc.Range($start, $start + $text.Length).Paragraphs.Item(1)
  $p.Range.Font.NameFarEast = '黑体'
  $p.Range.Font.Size = if ($level -eq 1) { 15 } else { 13 }
  $p.Range.Bold = 1
  $p.Range.ParagraphFormat.SpaceBefore = 10
  $p.Range.ParagraphFormat.SpaceAfter = 5
  return $p
}

function Add-Body($doc, [string]$text, [switch]$Bold) {
  $start = $doc.Content.End - 1
  $insert = $doc.Range($start, $start)
  $insert.InsertAfter($text + "`r")
  $p = $doc.Range($start, $start + $text.Length).Paragraphs.Item(1)
  $p.Range.Font.Name = '宋体'
  $p.Range.Font.NameFarEast = '宋体'
  $p.Range.Font.Size = 10.5
  if ($Bold) { $p.Range.Bold = 1 }
  $p.Range.ParagraphFormat.SpaceAfter = 4
  $p.Range.ParagraphFormat.LineSpacingRule = 1
  $p.Range.ParagraphFormat.KeepTogether = -1
  return $p
}

function Add-ReportTable($doc, $word, [string[]]$headers, [object[]]$rows, [double[]]$widths = $null) {
  $table = $doc.Tables.Add($doc.Range($doc.Content.End - 1, $doc.Content.End - 1), $rows.Count + 1, $headers.Count)
  for ($c = 1; $c -le $headers.Count; $c++) { Set-CellText $table.Cell(1, $c) $headers[$c - 1] }
  for ($r = 0; $r -lt $rows.Count; $r++) {
    for ($c = 0; $c -lt $headers.Count; $c++) { Set-CellText $table.Cell($r + 2, $c + 1) ([string]$rows[$r][$c]) }
  }
  Apply-TableStyle $table $word
  if ($widths) {
    for ($c = 1; $c -le $widths.Count; $c++) { $table.Columns.Item($c).PreferredWidth = $word.CentimetersToPoints($widths[$c - 1]) }
  }
  $doc.Paragraphs.Add() | Out-Null
  return $table
}

$word = $null
$sourceDoc = $null
$revDoc = $null
$reportDoc = $null
try {
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $word.DisplayAlerts = 0

  # Open the in-use source read-only and save a separate copy.
  $sourceDoc = $word.Documents.Open($sourcePath, $false, $true)
  $sourceDoc.SaveAs2($revisionPath, 16)
  $sourceDoc.Close(0)
  $sourceDoc = $null
  $revDoc = $word.Documents.Open($revisionPath, $false, $false)
  $word.UserName = '严格证据审计'
  $word.UserInitials = 'QA'

  Replace-Paragraph $revDoc `
    '脑肿瘤 MRI 病灶的准确自动分割对定量影像分析具有重要临床价值。然而，现有深度学习模型常将多通道输入、数据增强、网络结构与预训练组合使用，各改进组件的独立贡献缺乏严格验证。针对此问题，本文基于 kaggle_3m 数据集，在严格防止数据泄漏的统一患者级划分下构建了对照实验。以普通 U-Net 为基线，通过多随机种子配对消融，独立评估了三通道输入、轻量增强、ResNet34 编码器和 ImageNet 预训练的效用。结果表明，完整方案 M4-P 在测试集上取得了最优综合表现（Positive Macro IoU 0.7664，Positive Dice 0.8425）。其中，ResNet34 编码器提供了最稳定的正向贡献，其余组件呈现辅助性正向趋势。结合定性误差分析，本研究为脑肿瘤分割模型的组件设计与优化提供了可靠依据。' `
    '脑肿瘤 MRI 病灶的自动分割可为定量影像分析提供基础。本文基于 kaggle_3m 数据集，在固定患者级划分下比较普通 U-Net 与 ResNet34-U-Net，并通过随机种子 42、123 和 2026 的配对消融考察三通道输入、轻量增强、ResNet34 编码器和 ImageNet 预训练。M4-P 的测试集 Positive Macro IoU 和 Positive Dice 分别为 0.7664±0.0086 和 0.8425±0.0101，在当前八种配置中均值最高。ResNet34 编码器配置的三个种子差值均为正，患者级描述性 bootstrap 区间未跨 0；三通道输入、轻量增强和预训练仅呈弱正向平均趋势，尚未形成跨种子与患者一致的独立增益。本结果仅适用于当前固定测试集和实验协议。' `
    '摘要已按原始运行结果和10名测试患者的证据边界重写；避免“严格防止”“最优综合表现”“可靠依据”等超出当前内部验证范围的表述。'

  Replace-Paragraph $revDoc `
    '完整方案取得整体改善，其中 ResNet34 贡献最稳定，其余组件呈现辅助性正向信号。' `
    '在当前固定测试集上，M4-P 的两项主要阳性切片重叠指标均值最高；ResNet34 编码器配置的配对差值方向最一致，其余组件尚未形成稳定的独立增益证据。' `
    '贡献点改为与表3和表4能够直接支持的结果一致。'

  Replace-Paragraph $revDoc `
    '图像统一缩放至 256×256 并归一化；训练集采用随机水平翻转和最大 ±5° 旋转，图像与掩膜同步变换；掩膜以阈值 128 二值化。单通道配置提取 FLAIR，三通道配置使用完整通道，缺失序列由 FLAIR 填充。针对病灶前景与背景的类别不平衡，训练时采用正掩膜采样与空掩膜平衡。' `
    '图像统一缩放至 256×256，并按 mean=0.5、std=0.5 归一化；轻量增强配置对训练图像与掩膜同步实施概率为 0.5 的随机水平翻转和最大 ±5° 旋转，掩膜以阈值 128 二值化。单通道配置读取原始 RGB 图像的绿色通道作为 FLAIR，三通道配置直接读取原始 RGB 三通道。训练集启用空掩膜平衡，并将正掩膜采样比例设置为 0.40。' `
    '严重事实修正：src/brain_tumor_seg/data.py 只实现 flair_green（读取绿色通道）和 rgb_multimodal（直接读取 RGB）两种路径，没有检测缺失序列或以 FLAIR 填充的代码。其余参数来自24组 resolved_config.json。'

  Replace-Paragraph $revDoc `
    '八种配置均使用随机种子 42、123 和 2026，在同一固定清单与训练协议下独立训练。本文汇总全部重复实验的均值和样本标准差，不选择单次最高结果；组件效应按相同种子配对计算，以削弱不同随机状态对模型差异的干扰。运行记录复核表明，各实验清单一致，重新汇总结果与表3、表4相符。训练日志还显示所有模型均能平稳收敛，验证集选择与早停策略按预设协议执行。' `
    '八种配置均使用随机种子 42、123 和 2026，在同一固定清单与训练协议下独立训练。本文汇总全部重复实验的均值和样本标准差，不选择单次最高结果；组件效应按相同种子配对计算。24 组 test_metrics.json 的 manifest_sha256 均为 6aee075d9ea8b4a41cdefc372e381a55ec3a099acfaf16002dbde2cd7d5051e7，每次测试均包含 525 张切片（173 张阳性、352 张空切片），且均使用验证集 Positive Macro IoU 对应的最佳检查点和 0.5 阈值。训练曲线总体下降，但不同配置和随机种子的最佳 epoch 与验证波动存在差异，因此不将其概括为“所有模型均平稳收敛”。' `
    '证据来自24组 training_summary.json、test_metrics.json 与 resolved_config.json；原“所有模型均平稳收敛”缺少统一、预定义的判定标准，已改为可核验的日志事实。'

  Replace-Paragraph $revDoc `
    '如表3所示，完整方案 M4-P 在主要阳性切片重叠指标上取得最高均值，相对 E0 的配对变化方向一致，说明多通道输入、增强、残差编码器和预训练的联合使用改善了病灶区域的整体覆盖与轮廓匹配。若干 ResNet34 系列配置与完整方案接近，进一步表明编码器结构是总体优势的重要来源，而输入与训练策略主要发挥补充作用。' `
    '如表3所示，M4-P 的 Positive Macro IoU 和 Positive Dice 均值在当前八种配置中最高；相对 E0 的 Positive Macro IoU 在三个训练种子下均提高。该结果支持“完整配置组合在本固定测试集上的主要阳性切片重叠指标更高”，但不能把总体差异直接分摊给其中每个组件。ResNet34 配对对照的三个种子差值均为正，且患者级描述性 bootstrap 区间未跨 0；其余组件的种子方向不完全一致，患者级区间均跨 0。' `
    '将“说明联合使用改善”收紧为当前固定测试集上的描述性结果。组件归因必须依据表4中的严格配对，而不能由完整方案与基线的总体差异反推。'

  Replace-Paragraph $revDoc `
    '表4显示，M4-P 综合表现最佳，ResNet34 编码器提供了最一致、最清晰的正向配对贡献。更强的层级表征和多尺度编码能力有助于同时保留病灶语义与边界信息，是完整方案中的核心结构改进。该对照改变的是编码器整体配置，因而结论指向 ResNet34 方案的综合作用，而不把收益归因于某一内部模块。' `
    '表4显示，在本文预设的 Positive Macro IoU 对照中，ResNet34 编码器配置的三个训练种子差值均为正，10 名测试患者的平均差值也均为正；其患者级描述性 bootstrap 95% 区间为[+0.0331,+0.1071]。这些结果支持其为当前四项组件中方向最一致的正向信号。残差和多尺度表征是相关研究提出的设计动机[5]，但本实验没有单独操纵内部残差连接、深度或特征金字塔，不能证明具体机制。' `
    '机制归因降调：本文对照同时改变整个编码器配置，只能得出 ResNet34 配置的描述性贡献；[5]支持残差/多尺度的设计动机，不是本文机制的直接证据。'

  Replace-Paragraph $revDoc `
    '三通道输入、轻量增强和 ImageNet 预训练的平均变化均为正，说明三者已呈现辅助性积极信号。三通道输入可在部分状态和病例中利用互补成像信息，但缺失序列填充带来的通道冗余限制了优势释放；轻量几何扰动提供了一定正则化支持，进一步优化变换类型与强度有望提高稳定性；预训练在多数配对中保持正向，提示良好初始化能够辅助优化，其收益仍受自然图像与 MRI 域差异、通道映射和微调策略影响。' `
    '三通道输入、轻量增强和 ImageNet 预训练的平均差值均为正，但三者都只有 2/3 个训练种子为正，患者级描述性 bootstrap 95% 区间也均跨 0。因此，当前证据仅支持“未形成稳定独立增益的弱正向趋势”。多模态互补、增强正则化以及迁移学习初始化是文献中的一般动机[3][6]，不能作为本文差异的已证实机制；尤其不能用当前代码中不存在的“缺失序列填充”解释结果。' `
    '删除无实现依据的缺失序列填充解释，并按2/3正向种子和患者级区间跨0收紧结论。文献仅支持技术动机。'

  Replace-Paragraph $revDoc `
    '组件贡献不能简单相加。输入、增强、编码器和初始化可能存在互补、冗余或条件依赖，配对效应也会随训练状态和病例变化。当前证据表明 ResNet34 构成稳定核心，其余三项组件显示与完整方案一致的正向倾向，并提供继续开发的空间。完整多模态数据、增强策略搜索及医学影像预训练与分层微调，可用于验证并放大这些信号。' `
    '组件效应不能简单相加，因为各比较对应不同配置背景。当前证据支持 ResNet34 编码器配置为方向最一致的正向信号；三通道输入、轻量增强和 ImageNet 预训练尚未形成跨训练种子与患者均一致的独立增益。后续应通过更大患者样本、预注册对照、外部验证以及针对输入构成和预训练策略的独立实验重新检验这些组件。' `
    '“稳定核心”和“与完整方案一致”可能被理解为统计确认，已改为与现有3个种子、10名患者及描述性bootstrap相匹配的表述。'

  Replace-Paragraph $revDoc `
    '此外，训练随机性与患者差异分别反映优化波动和病例不确定性，定量与定性结果均表现出一定的病例依赖性。多随机种子与患者级配对汇总表明，完整方案和 ResNet34 的改善最为一致；其他组件的收益可能受病灶尺度、图像质量与通道完整性影响，后续应扩大患者样本并开展预先定义的亚组分析。' `
    '此外，训练随机性与患者差异分别反映优化波动和病例不确定性。多随机种子与患者级配对汇总显示，完整方案相对 E0、ResNet34 配置相对 M0-AB 的差值方向最一致；其余组件的差值方向不一致。病灶尺度、图像质量或原始 RGB 通道构成是否影响组件收益，当前实验尚未进行预先定义的亚组检验，需在扩大患者样本后另行验证。' `
    '删除“通道完整性影响收益”的未经检验推断；当前数据清单不提供可用于该结论的缺失模态字段。'

  Replace-Paragraph $revDoc `
    '本文在固定数据清单和统一训练协议下比较二维脑肿瘤分割方案。M4-P 获得最佳综合表现，ResNet34 编码器贡献最稳定；三通道输入、轻量增强与预训练均呈辅助性正向趋势，为后续优化多模态信息与训练策略提供了依据。' `
    '本文在固定患者级清单和统一训练协议下比较二维脑肿瘤切片分割方案。M4-P 的 Positive Macro IoU 与 Positive Dice 均值在当前八种配置中最高；ResNet34 编码器配置呈现方向最一致的正向配对信号。三通道输入、轻量增强与 ImageNet 预训练的平均差值为正，但跨训练种子和患者的方向不一致，尚不足以认定为稳定独立增益。' `
    '结论按最严格证据口径重写，避免“最佳综合表现”和“贡献最稳定”超出已报告指标与有限患者样本。'

  Replace-Paragraph $revDoc `
    '本研究受限于单一公开数据源、二维架构与有限病例，尚未评估三维体积误差或接受影像科专家盲法评价；缺失序列填充也限制了完整多模态价值的判断。因此，现有结果主要反映受控实验中的方法差异。' `
    '本研究受限于单一公开数据源、二维架构与有限病例，尚未评估三维体积误差或接受影像科专家盲法评价。当前三通道配置直接使用公开 TIFF 文件中的 RGB 三通道，仓库未提供逐患者原始序列完整性字段，因此不能把结果解释为对完整、独立 MRI 模态组合的验证。现有结果主要反映当前通道构造和受控实验协议下的方法差异。' `
    '删除不存在的“缺失序列填充”事实，并明确当前仓库能够确认的输入边界。'

  Replace-Paragraph $revDoc `
    '未来将面向多中心独立数据开展验证，引入相邻切片或三维上下文，完善多模态与缺失模态建模，并优化增强策略、医学影像预训练和分层微调；同时结合专家盲评、边界修订成本与严重漏检误报分析，系统评价预测校准和不同患者亚组的稳定性，检验模型的真实临床辅助价值。' `
    '未来将面向多中心独立数据开展验证，引入相邻切片或三维上下文，并在具有明确序列元数据的数据上分别评估完整模态与缺失模态情形；同时优化增强、医学影像预训练和分层微调，并结合专家盲评、边界修订成本与严重漏检误报分析，评价预测校准和患者亚组稳定性。' `
    '未来工作与当前数据事实边界对齐，不再暗示本实验已经识别出具体缺失模态。'

  # Reference [8] citation limitation.
  Replace-Paragraph $revDoc `
    '多数模型改进以组合形式出现，整体性能提高并不能直接拆解输入、增强、结构和预训练的独立贡献；不同组件还可能产生互补或冗余。与此同时，医学影像中的相邻切片和同一受试者样本具有相关性，不恰当的交叉验证容易引发数据泄漏并夸大性能[8]。本文因此建立统一数据清单，在相同训练协议下采用多个随机种子，对四类组件进行严格配对消融，并结合总体指标、患者汇总与定性误差分析，提高内部比较的可解释性和可核查性。' `
    '多数模型改进以组合形式出现，整体性能提高并不能直接拆解输入、增强、结构和预训练的独立贡献；不同组件还可能产生互补或冗余。跨任务方法学研究表明，在同一受试者可产生多个相关观测或增强样本时，非受试者级划分可能造成数据泄漏并高估模型性能[8]；该研究对象为精神疾病计算机辅助诊断，并非脑肿瘤分割，因此本文仅将其作为受试者级划分原则的佐证。本文据此采用固定患者级清单和组件配对对照，以提高内部比较的可核查性。' `
    '[8]使用EEG与精神疾病诊断任务，不可直接作为脑肿瘤相邻切片泄漏的任务内证据；修订后明确其跨任务方法学性质。'

  # Add traceability comments to figures.
  Add-Comment-ToParagraph $revDoc '图1 M4-P 的 ResNet34-U-Net 模型结构' '图1为依据 src/brain_tumor_seg/model.py 中 ResNet34UNet 实现绘制的结构示意图，不是实验输出或性能证据。'
  Add-Comment-ToParagraph $revDoc '图2 E0与M4-P对比及典型失败案例' '图2案例可回溯到 seed42 的 evaluation/test/samples.csv 与 comparisons/test/LGG 输出：slice_45、slice_22、slice_23、slice_51 的IoU及混淆像素均已逐项复核。案例仅用于定性解释。'

  # Remove unused references [7], [11]-[14], retain all used items and renumber [15]->[11], [16]->[12].
  $unused = @(
    '[7] BERKLEY A, SAUERESSIG C, SHUKLA U, et al. Clinical capability of modern brain tumor segmentation models[J]. Medical Physics, 2023, 50(8): 4943-4959. DOI: 10.1002/mp.16321. PMID: 36847185.',
    '[11] VERMA A, YADAV A K. Brain tumor segmentation with deep learning: Current approaches and future perspectives[J]. Journal of Neuroscience Methods, 2025, 418: 110424. DOI: 10.1016/j.jneumeth.2025.110424. PMID: 40122469.',
    '[12] HADDADI AVVAL A, BANERJEE S, ZIELKE J, et al. Applications of artificial intelligence and advanced imaging in pediatric diffuse midline glioma[J]. Neuro-Oncology, 2025, 27(6): 1419-1433. DOI: 10.1093/neuonc/noaf058. PMID: 40037540.',
    '[13] FENG X, GHIMIRE K, KIM D D, et al. Brain Tumor Segmentation for Multi-Modal MRI with Missing Information[J]. Journal of Digital Imaging, 2023, 36(5): 2075-2087. DOI: 10.1007/s10278-023-00860-7. PMID: 37340197.',
    '[14] AHSAN R, SHAHZADI I, NAJEEB F, et al. Brain tumor detection and segmentation using deep learning[J]. MAGMA, 2025, 38(1): 13-22. DOI: 10.1007/s10334-024-01203-5. PMID: 39231857.'
  )
  foreach ($ref in $unused) {
    $p = Find-ExactParagraph $revDoc $ref
    $p.Range.Delete()
  }
  # Renumber all retained citations without cascading replacements.
  foreach ($p in $revDoc.Paragraphs) {
    $r = $p.Range.Duplicate; $r.End = $r.End - 1
    $text = $r.Text
    if ($text -match '\[(8|9|10|15|16)\]') {
      $newText = $text.Replace('[16]','[[REF_DATASET]]').Replace('[15]','[[REF_MAU]]').Replace('[10]','[[REF_EXPERT]]').Replace('[9]','[[REF_REVIEW]]').Replace('[8]','[[REF_LEAK]]')
      $newText = $newText.Replace('[[REF_DATASET]]','[11]').Replace('[[REF_MAU]]','[10]').Replace('[[REF_EXPERT]]','[9]').Replace('[[REF_REVIEW]]','[8]').Replace('[[REF_LEAK]]','[7]')
      $r.Text = $newText
    }
  }

  # Final reference comment and update fields.
  Add-Comment-ToParagraph $revDoc '[7] LEE H T, CHEON H R, LEE S H, et al. Risk of data leakage in estimating the diagnostic performance of a deep-learning-based computer-aided system for psychiatric disorders[J]. Scientific Reports, 2023, 13(1): 16633. DOI: 10.1038/s41598-023-43542-8. PMID: 37789047.' '真实性已核验，但研究对象为EEG精神疾病诊断；仅适合作为受试者级划分的一般方法学佐证。'
  if ($revDoc.Fields.Count -gt 0) { $revDoc.Fields.Update() | Out-Null }
  $revDoc.Save()

  # Build formal audit report.
  $reportDoc = $word.Documents.Add()
  $section = $reportDoc.Sections.Item(1)
  $section.PageSetup.TopMargin = $word.CentimetersToPoints(2.2)
  $section.PageSetup.BottomMargin = $word.CentimetersToPoints(2.2)
  $section.PageSetup.LeftMargin = $word.CentimetersToPoints(2.2)
  $section.PageSetup.RightMargin = $word.CentimetersToPoints(2.2)
  $titleStart = $reportDoc.Content.End - 1
  $titleInsert = $reportDoc.Range($titleStart, $titleStart)
  $titleText = '论文实验数据与参考文献严格审计报告'
  $titleInsert.InsertAfter($titleText + "`r")
  $title = $reportDoc.Range($titleStart, $titleStart + $titleText.Length).Paragraphs.Item(1)
  $title.Range.Font.NameFarEast = '黑体'; $title.Range.Font.Size = 20; $title.Range.Bold = 1
  $title.Alignment = 1
  Add-Body $reportDoc '审计对象：论文中文草稿框架_第三至六部分完善终稿_会议论文结构再精简版.docx' | Out-Null
  Add-Body $reportDoc '审计日期：2026年8月13日　　证据标准：无直接证据即降调、删除或补充来源' | Out-Null

  Add-Heading $reportDoc '一、审计结论摘要' 1 | Out-Null
  Add-Body $reportDoc '总体结论：论文表1、表3和表4的实验数值均可由当前仓库中的数据清单与24组原始运行结果复现；8种配置的关键组件矩阵与论文表2一致。发现1项严重事实错误、若干证据强度与措辞不匹配问题，以及5条未被当前精简正文使用的参考文献。修订稿已逐项处理。' -Bold | Out-Null
  $summaryRows = @(
    @('严重','“缺失序列由FLAIR填充”无代码或清单证据','删除并改为实际实现：绿色通道FLAIR或直接RGB读取'),
    @('重要','“所有模型平稳收敛”无统一判据','改为曲线总体下降但最佳epoch与验证波动不同'),
    @('重要','对三通道、增强、预训练的独立贡献表述偏强','改为弱正向趋势，明确2/3种子为正且患者级区间跨0'),
    @('重要','残差/多尺度机制被写成本文已证实原因','改为文献设计动机，不作本文因果结论'),
    @('一般','[8]跨任务引用未限定研究对象','明确为EEG精神疾病诊断的通用方法学佐证'),
    @('一般','5条文献未在正文引用','修订稿删除并重编号；本报告保留真实性核验')
  )
  Add-ReportTable $reportDoc $word @('等级','问题','处置') $summaryRows @(2.0,7.0,7.0) | Out-Null

  Add-Heading $reportDoc '二、实验数据证据矩阵' 1 | Out-Null
  Add-Body $reportDoc '独立读取24组 resolved_config.json、training_summary.json、test_metrics.json、environment.json 与 evaluation/test/samples.csv。未依赖既有Markdown汇总作为计算输入。' | Out-Null
  $evidenceRows = @(
    @('清洁清单','104名患者；3,629张切片；排除6名患者/300张切片','splits/kaggle_3m_multimodal_only_seed42.meta.json','一致'),
    @('患者级划分','85/9/10名患者；2,619/485/525张切片','split meta + manifest','一致'),
    @('正/空掩膜','训练935/1,684；验证167/318；测试173/352','split meta + 24组test_metrics','一致'),
    @('运行清单哈希','6aee075d...d5051e7，24/24一致','24组test_metrics','一致'),
    @('测试协议','最佳验证Positive Macro IoU检查点；阈值0.5','24组training_summary/test_metrics','一致'),
    @('实验环境','RTX 5060 Laptop GPU；CUDA 12.8；PyTorch 2.11.0+cu128','24组environment.json','一致')
  )
  Add-ReportTable $reportDoc $word @('项目','独立核验结果','证据','结论') $evidenceRows @(2.6,6.0,5.1,1.8) | Out-Null

  Add-Heading $reportDoc '三、表3数值独立复算' 1 | Out-Null
  Add-Body $reportDoc '均值与样本标准差均从3个训练种子的原始 test_metrics.json 重新计算。以下值与论文显示精度完全一致。' | Out-Null
  $metricRows = @()
  foreach ($model in @('E0','E1-A','E2-B','M0-AB','M4-NP','M4-P−A','M4-P−B','M4-P')) {
    $s = $audit.summaries.$model
    $metricRows += ,@(
      $model,
      ('{0:F4}±{1:F4}' -f $s.positive_macro_iou.mean,$s.positive_macro_iou.sd),
      ('{0:F4}±{1:F4}' -f $s.positive_macro_dice.mean,$s.positive_macro_dice.sd),
      ('{0:F4}±{1:F4}' -f $s.micro_iou.mean,$s.micro_iou.sd),
      ('{0:F4}±{1:F4}' -f $s.micro_precision.mean,$s.micro_precision.sd),
      ('{0:F4}±{1:F4}' -f $s.micro_recall.mean,$s.micro_recall.sd),
      ('{0:F2}%±{1:F2}%' -f ($s.empty_slice_false_positive_rate.mean*100),($s.empty_slice_false_positive_rate.sd*100))
    )
  }
  Add-ReportTable $reportDoc $word @('模型','Pos IoU','Pos Dice','Micro IoU','Precision','Recall','空切片误报') $metricRows | Out-Null

  Add-Heading $reportDoc '四、表4配对消融与不确定性' 1 | Out-Null
  $effectRows = @()
  foreach ($name in @('完整方案','三通道输入 A','轻量增强 B','ResNet34 结构 C','ImageNet 预训练 D')) {
    $e = $audit.effects.$name
    $effectRows += ,@(
      $name,$e.comparison,
      ('{0:+0.0000;-0.0000;0.0000}±{1:F4}' -f $e.seed_mean,$e.seed_sd),
      ("$($e.positive_seeds)/3"),
      ('{0:+0.0000;-0.0000;0.0000}' -f $e.patient_mean),
      ('[{0:+0.0000;-0.0000;0.0000},{1:+0.0000;-0.0000;0.0000}]' -f $e.ci95[0],$e.ci95[1])
    )
  }
  Add-ReportTable $reportDoc $word @('组件','比较','Seed差值','正向种子','患者均差','患者95% CI') $effectRows | Out-Null
  Add-Body $reportDoc '解释边界：bootstrap以10名固定测试患者为重采样单位，仅用于描述患者间不确定性。三个训练种子不是新增独立患者，不能替代患者级交叉验证或外部验证。A、B、D的患者级区间均跨0，不应表述为已确认的稳定独立贡献。' -Bold | Out-Null

  Add-Heading $reportDoc '五、配置与方法一致性' 1 | Out-Null
  $configRows = @(
    @('E0','绿色通道FLAIR','U-Net','无','无','一致'),
    @('E1-A','RGB三通道','U-Net','无','无','一致'),
    @('E2-B','绿色通道FLAIR','U-Net','翻转+±5°','无','一致'),
    @('M0-AB','RGB三通道','U-Net','翻转+±5°','无','一致'),
    @('M4-NP','RGB三通道','ResNet34-U-Net','翻转+±5°','无','一致'),
    @('M4-P−A','绿色通道FLAIR','ResNet34-U-Net','翻转+±5°','ImageNet','一致'),
    @('M4-P−B','RGB三通道','ResNet34-U-Net','无','ImageNet','一致'),
    @('M4-P','RGB三通道','ResNet34-U-Net','翻转+±5°','ImageNet','一致')
  )
  Add-ReportTable $reportDoc $word @('模型','输入','结构','增强','预训练','核验') $configRows | Out-Null
  Add-Body $reportDoc '严重问题：data.py 的 rgb_multimodal 分支仅执行 image_file.convert("RGB")；flair_green 分支读取绿色通道。代码和清单均无“缺失模态检测”或“由FLAIR填充缺失序列”实现，故原文该陈述及相关因果解释不成立。' -Bold | Out-Null

  Add-Heading $reportDoc '六、图像与案例溯源' 1 | Out-Null
  $figureRows = @(
    @('图1','结构示意图','可回溯至model.py的ResNet34-U-Net实现；不是实验结果','保留，批注明确性质'),
    @('图2案例1','slice_45：E0 IoU 0.1259；M4-P IoU 0.9658','两模型seed42 samples.csv及comparison图','已核实'),
    @('图2案例2','slice_22：E0/M4-P IoU 0.4171/0.1590；M4-P FP=877','两模型seed42 samples.csv及comparison图','已核实'),
    @('图2漏检','slice_23：真实前景265像素；M4-P无预测前景','M4-P seed42 samples.csv及comparison图','已核实'),
    @('图2空切片误报','slice_51：M4-P FP=4,584，占65,536像素约7.0%','M4-P seed42 samples.csv及comparison图','已核实')
  )
  Add-ReportTable $reportDoc $word @('对象','论文内容','证据','结论') $figureRows @(2.3,5.6,5.8,2.2) | Out-Null

  Add-Heading $reportDoc '七、16条参考文献真实性核验' 1 | Out-Null
  $refRows = @(
    @('[1]','39074952','10.1002/jmri.29543','真实；元数据一致','正文直接使用'),
    @('[2]','38011781','10.1016/j.compmedimag.2023.102313','真实；元数据一致','正文直接使用'),
    @('[3]','39104626','10.3389/fbioe.2024.1392807','真实；元数据一致','正文直接使用'),
    @('[4]','37967585','10.1016/j.radonc.2023.110007','真实；研究脑转移瘤','仅作评价背景'),
    @('[5]','37981627','10.1007/s11517-023-02965-1','真实；元数据一致','支持设计动机'),
    @('[6]','40745592','10.1186/s12880-025-01837-4','真实；元数据一致','支持迁移学习路线'),
    @('[7]','36847185','10.1002/mp.16321','真实；元数据一致','当前正文未引用，删除'),
    @('[8]','37789047','10.1038/s41598-023-43542-8','真实；EEG精神疾病诊断','限缩为跨任务方法学'),
    @('[9]','38102956','10.1177/15353702231214259','真实；元数据一致','正文直接使用'),
    @('[10]','38197800','10.1148/ryai.220231','真实；专家中心评价','正文直接使用'),
    @('[11]','40122469','10.1016/j.jneumeth.2025.110424','真实；元数据一致','当前正文未引用，删除'),
    @('[12]','40037540','10.1093/neuonc/noaf058','真实；儿童弥漫中线胶质瘤','当前正文未引用，删除'),
    @('[13]','37340197','10.1007/s10278-023-00860-7','真实；缺失MRI序列方法','当前正文未引用，删除'),
    @('[14]','39231857','10.1007/s10334-024-01203-5','真实；检测+分割','当前正文未引用，删除'),
    @('[15]','38124563','10.3934/mbe.2023907','真实；元数据一致','保留并重编号[10]'),
    @('[16]','无','Kaggle官方数据页','页面真实；许可标为CC BY-NC-SA 4.0','保留并重编号[11]')
  )
  Add-ReportTable $reportDoc $word @('原编号','PMID','DOI/来源','真实性','正文处置') $refRows | Out-Null

  Add-Heading $reportDoc '八、正文引述支持度矩阵' 1 | Out-Null
  $citationRows = @(
    @('引言：临床价值、人工勾画困难','[1][2]','直接/较强支持','保留'),
    @('相关研究：多模态MRI互补信息','[3]','直接支持一般背景','保留；不声称本文RGB必为完整独立模态'),
    @('U-Net及评价指标背景','[4][9][10]','部分支持','[4]限定为脑转移瘤；不外推本文LGG性能'),
    @('残差、多尺度结构动机','[5]','直接支持文献自身设计','改为设计动机；不作本文因果证明'),
    @('迁移学习/预训练路线','[6]','直接支持路线','不证明本文ImageNet预训练稳定有效'),
    @('受试者级划分避免泄漏','[8]','跨任务部分支持','明确研究为EEG精神疾病诊断'),
    @('混合注意力结构','原[15]→[10]','直接支持文献自身方法','保留'),
    @('数据集来源与许可','原[16]→[11]','官方页面支持','保留')
  )
  Add-ReportTable $reportDoc $word @('正文命题','引用','支持度','最终处理') $citationRows @(5.4,2.3,4.1,4.0) | Out-Null

  Add-Heading $reportDoc '九、最终修订清单与验收结论' 1 | Out-Null
  Add-Body $reportDoc '1. 删除无实现证据的缺失模态填充陈述及相关结果解释。' | Out-Null
  Add-Body $reportDoc '2. 将“所有模型平稳收敛”、组件稳定贡献和具体机制归因改为可由日志与配对结果直接支持的限定表述。' | Out-Null
  Add-Body $reportDoc '3. 对[4]和[8]明确任务边界；把残差、多模态、注意力与迁移学习文献定位为技术背景或设计动机。' | Out-Null
  Add-Body $reportDoc '4. 删除5条未在精简正文使用的真实文献，并将保留条目连续重编号为[1]—[11]。' | Out-Null
  Add-Body $reportDoc '5. 在修订稿关键修改和图像处插入真实Word批注，便于逐处复核。' | Out-Null
  Add-Body $reportDoc '验收结论：修订后正文中的实验数值均可回溯至配置、清单、运行结果或逐切片记录；参考文献均真实存在，正文保留的引述均按原文证据范围进行了限定。结果仍仅适用于当前固定患者级划分和10名测试患者，不构成外部泛化或临床有效性证据。' -Bold | Out-Null

  $reportDoc.SaveAs2($reportPath, 16)
  $reportDoc.Close(0); $reportDoc = $null
  $revDoc.Close(0); $revDoc = $null
  Write-Output $revisionPath
  Write-Output $reportPath
}
finally {
  if ($reportDoc) { $reportDoc.Close(0) }
  if ($revDoc) { $revDoc.Close(0) }
  if ($sourceDoc) { $sourceDoc.Close(0) }
  if ($word) { $word.Quit() }
  [GC]::Collect(); [GC]::WaitForPendingFinalizers()
}
