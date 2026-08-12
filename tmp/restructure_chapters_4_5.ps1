param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [Parameter(Mandatory = $true)]
    [string]$Output
)

Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression

$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$outputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
Copy-Item -LiteralPath $sourcePath -Destination $outputPath -Force

$archive = [System.IO.Compression.ZipFile]::Open($outputPath, [System.IO.Compression.ZipArchiveMode]::Update)
try {
    $entry = $archive.GetEntry('word/document.xml')
    if (-not $entry) { throw 'word/document.xml not found' }
    $reader = [System.IO.StreamReader]::new($entry.Open())
    try { $xmlText = $reader.ReadToEnd() } finally { $reader.Dispose() }

    $xml = [System.Xml.XmlDocument]::new()
    $xml.PreserveWhitespace = $true
    $xml.LoadXml($xmlText)
    $wUri = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    $ns = [System.Xml.XmlNamespaceManager]::new($xml.NameTable)
    $ns.AddNamespace('w', $wUri)

    function Get-ParagraphText([System.Xml.XmlNode]$Paragraph) {
        $nodes = $Paragraph.SelectNodes('.//w:t', $ns)
        return (($nodes | ForEach-Object { $_.InnerText }) -join '')
    }

    function Find-Paragraph([string]$ExactText) {
        $matches = @($xml.SelectNodes('//w:body//w:p', $ns) | Where-Object { (Get-ParagraphText $_) -eq $ExactText })
        if ($matches.Count -ne 1) { throw "Expected one paragraph for: $ExactText; found $($matches.Count)" }
        return $matches[0]
    }

    function Set-ParagraphText([System.Xml.XmlNode]$Paragraph, [string]$Text) {
        $children = @($Paragraph.ChildNodes)
        foreach ($child in $children) {
            if (-not ($child.LocalName -eq 'pPr' -and $child.NamespaceURI -eq $wUri)) {
                [void]$Paragraph.RemoveChild($child)
            }
        }
        $run = $xml.CreateElement('w', 'r', $wUri)
        $textNode = $xml.CreateElement('w', 't', $wUri)
        if ($Text.StartsWith(' ') -or $Text.EndsWith(' ')) {
            $spaceAttr = $xml.CreateAttribute('xml', 'space', 'http://www.w3.org/XML/1998/namespace')
            $spaceAttr.Value = 'preserve'
            [void]$textNode.Attributes.Append($spaceAttr)
        }
        $textNode.InnerText = $Text
        [void]$run.AppendChild($textNode)
        [void]$Paragraph.AppendChild($run)
    }

    function Replace-Paragraph([string]$OldText, [string]$NewText) {
        $p = Find-Paragraph $OldText
        Set-ParagraphText $p $NewText
        return $p
    }

    function Replace-SectionBody([string]$HeadingText, [string]$NextHeadingText, [string[]]$NewParagraphs) {
        $heading = Find-Paragraph $HeadingText
        $nextHeading = Find-Paragraph $NextHeadingText
        $bodyNodes = [System.Collections.Generic.List[System.Xml.XmlNode]]::new()
        $cursor = $heading.NextSibling
        while ($cursor -and $cursor -ne $nextHeading) {
            $next = $cursor.NextSibling
            if ($cursor.LocalName -eq 'p' -and $cursor.NamespaceURI -eq $wUri) { $bodyNodes.Add($cursor) }
            $cursor = $next
        }
        if ($bodyNodes.Count -lt 1) { throw "No body paragraph found after $HeadingText" }
        $template = $bodyNodes[0].CloneNode($true)
        foreach ($node in $bodyNodes) { [void]$node.ParentNode.RemoveChild($node) }
        foreach ($text in $NewParagraphs) {
            $newP = $template.CloneNode($true)
            Set-ParagraphText $newP $text
            [void]$nextHeading.ParentNode.InsertBefore($newP, $nextHeading)
        }
    }

    function Delete-Paragraph([string]$ExactText) {
        $p = Find-Paragraph $ExactText
        [void]$p.ParentNode.RemoveChild($p)
    }

    # 第四部分：保留数值与可复核观察，将解释性讨论移至第五部分。
    Replace-Paragraph '可写入讨论的谨慎结论是：完整方案相对 E0 的提升在三个训练种子中方向一致，ResNet34 结构的配对提升也在三个种子中为正；三通道输入、轻量增强和 ImageNet 预训练的平均差值虽为正，但区间跨越 0 或种子方向不完全一致，暂不宜写成稳定、独立的确定性贡献。' '严格配对结果显示，M4-P−E0 与 ResNet34 结构对照在三个训练随机种子下的差值均为正；三通道输入、轻量增强和 ImageNet 预训练的平均差值为正，但患者级 bootstrap 95% 区间均跨越 0。各组件的作用及其不确定性在第 5.2 节集中讨论。' | Out-Null

    Replace-Paragraph '从验证集曲线可以看出，M4-P、M4-P−A、M4-P−B和M4-NP在训练早期即达到较高的 Positive Macro IoU，随后主要表现为小幅波动。M4-P三个随机种子的最佳验证 Positive Macro IoU为0.7547±0.0058，对应最佳epoch分别为25、25和34，表明完整模型能够在较早阶段达到相对稳定的验证性能。M4-P−A的最佳验证指标为0.7413±0.0007，跨种子波动最小；M4-NP和M4-P−B分别为0.7278±0.0272和0.7303±0.0046。上述现象说明ResNet34系列整体收敛较快，但预训练、输入方式和数据增强对不同随机种子的影响并不完全一致。' 'M4-P、M4-P−A、M4-P−B 和 M4-NP 在训练早期达到较高的验证集 Positive Macro IoU，随后以小幅波动为主。M4-P 三个随机种子的最佳验证指标为 0.7547±0.0058，对应最佳 epoch 分别为 25、25 和 34；M4-P−A、M4-NP 和 M4-P−B 的最佳验证指标分别为 0.7413±0.0007、0.7278±0.0272 和 0.7303±0.0046。' | Out-Null

    Replace-Paragraph '普通U-Net相关配置的收敛差异更明显。E0虽然训练集IoU持续升高、训练损失持续下降，但最佳训练指标与最佳验证指标的平均差距约为0.3120，明显大于M4-P的0.0573，提示E0存在更突出的训练—验证泛化差距。E2-B的最佳epoch分别为48、24和111，且曲线阴影较宽，反映其收敛速度对随机初始化较为敏感。M0-AB的平均训练—验证差距约为0.0544，与M4-P接近，说明三通道输入与轻量增强的组合有助于缩小该差距，但其最佳验证指标仍低于ResNet34系列。' '普通 U-Net 相关配置的曲线差异较大。E0 的最佳训练指标与最佳验证指标的平均差距为 0.3120，M4-P 和 M0-AB 的对应差距分别为 0.0573 和 0.0544。E2-B 三个随机种子的最佳 epoch 分别为 48、24 和 111，且曲线标准差带较宽。上述曲线现象的含义在第 5.4 节讨论。' | Out-Null

    Replace-Paragraph '训练损失整体下降而验证损失在中后期趋于平台或出现波动，说明继续降低训练误差不一定带来验证性能提升。因此，本文使用验证集 Positive Macro IoU选择最佳检查点，并配合早停策略，而不使用最后一个epoch作为最终模型。需要指出的是，图中均值和标准差仅描述三个训练随机种子下的优化波动，不能替代基于独立患者样本的统计推断；模型优劣仍应结合固定测试集结果、患者级配对分析及后续外部验证综合判断。' '各配置的训练损失总体下降，验证损失在中后期趋于平台或出现波动。按照预设协议，本文依据验证集 Positive Macro IoU 选择最佳检查点并采用早停，而不直接使用最后一个 epoch。图中均值和标准差仅汇总三个训练随机种子的曲线。' | Out-Null

    Replace-Paragraph '如图 4-2(a) 所示，样本 TCGA_DU_5872_19950223__slice_45 的预测区域与真实病灶高度重合，IoU、Precision 和 Recall 分别为 0.9658、0.9801 和 0.9851，仅在病灶边缘存在少量 FP 与 FN 像素，说明模型能够较准确地恢复边界清晰、面积较大的病灶。图 4-2(b) 展示显著过分割样本 TCGA_DU_6408_19860521__slice_22：模型保留了 0.7652 的 Recall，但在真实小病灶周围产生多个离散假阳性区域，FP 达 877 像素，Precision 降至 0.1671，IoU 仅为 0.1590。图 4-2(c) 为欠分割样本 TCGA_HT_7616_19940813__slice_15，其 Precision 为 0.9908，但 Recall 仅为 0.3106；预测几乎未扩展到病灶的大部分区域，共有 959 个 FN 像素。上述案例表明，相近或较高的单项 Precision/Recall 可能对应完全不同的空间错误模式，因此需要结合 IoU、混淆像素数和叠加图共同判断。' '图 4-2(a) 中，样本 TCGA_DU_5872_19950223__slice_45 的 IoU、Precision 和 Recall 分别为 0.9658、0.9801 和 0.9851，误差主要分布在病灶边缘。图 4-2(b) 的过分割样本 TCGA_DU_6408_19860521__slice_22 的 Recall 为 0.7652，但 FP 为 877 个像素，Precision 和 IoU 分别为 0.1671 和 0.1590。图 4-2(c) 的欠分割样本 TCGA_HT_7616_19940813__slice_15 的 Precision 为 0.9908、Recall 为 0.3106，FN 为 959 个像素。' | Out-Null

    Replace-Paragraph '图 4-3(a) 展示完全漏检样本 TCGA_HT_7881_19981015__slice_23。该切片真实病灶包含 265 个像素，但模型未预测出任何前景，IoU 和 Recall 均为 0。在种子 42 的 173 张阳性测试切片中，共有 3 张出现完全漏检。进一步按真实病灶面积将阳性切片分为四个近似等量组后，最小病灶组包含 43 张切片，病灶面积为 37～790 像素，其平均 IoU 和平均 Recall 分别为 0.5782 和 0.7369；3张完全漏检切片全部位于该组。相比之下，其余三个面积组的平均 IoU 分别为 0.7784、0.8834 和 0.8593。该描述性结果提示，小病灶和位于病灶边缘层面的切片是当前模型的主要薄弱环节，但由于切片来自同一患者且未进行专门的显著性检验，不能将面积差异解释为严格的因果关系。' '图 4-3(a) 的完全漏检样本 TCGA_HT_7881_19981015__slice_23 含 265 个真实病灶像素，模型未预测出前景，IoU 和 Recall 均为 0。种子 42 的 173 张阳性测试切片中有 3 张完全漏检。按真实病灶面积划分为四个近似等量组后，最小病灶组包含 43 张切片，面积范围为 37～790 个像素，平均 IoU 和 Recall 分别为 0.5782 和 0.7369，3 张完全漏检切片均位于该组；其余三组的平均 IoU 分别为 0.7784、0.8834 和 0.8593。' | Out-Null

    Replace-Paragraph '图 4-3(b) 展示空切片误报样本 TCGA_FG_8189_20030516__slice_51。其真实掩膜为空，但模型将 4584 个像素判定为肿瘤，占整张 256×256 图像的约 7.0%，说明模型可能将正常组织的局部高信号或强度异常误识别为病灶。种子 42 的 352 张空切片中有 55 张出现至少一个假阳性像素，空切片误报率为 15.63%。虽然该次运行的空切片平均预测前景比例仅为 0.0906%，少数严重误报仍会显著影响实际使用的可靠性，因此不能仅依据整体 Accuracy 或 Specificity 较高就认为背景误检问题已经解决。' '图 4-3(b) 的样本 TCGA_FG_8189_20030516__slice_51 真实掩膜为空，模型预测了 4584 个前景像素，约占 256×256 图像的 7.0%。种子 42 的 352 张空切片中有 55 张出现至少一个假阳性像素，空切片误报率为 15.63%，空切片平均预测前景比例为 0.0906%。' | Out-Null

    Replace-Paragraph '为避免仅依据总体均值解释模型差异，图 4-4 对 E0 基线与 M4-P 在相同测试切片上的输出进行比较。在案例 1（TCGA_DU_5872_19950223__slice_45）中，E0 仅覆盖病灶的一小部分，IoU 为 0.1259，而 M4-P 的 IoU 达到 0.9658，说明完整方案在该样本上明显改善了病灶覆盖和边界一致性。然而，在案例 2（TCGA_DU_6408_19860521__slice_22）中，E0 与 M4-P 的 Recall 均为 0.7652，但 M4-P 的 FP 像素由 192 增至 877，IoU 由 0.4171 降至 0.1590。该结果说明，M4-P 的总体平均性能更好并不意味着每个患者或每张切片都获得改善，模型增益具有明显的病例依赖性。' '图 4-4 比较 E0 与 M4-P 在相同测试切片上的输出。案例 1（TCGA_DU_5872_19950223__slice_45）中，两模型的 IoU 分别为 0.1259 和 0.9658。案例 2（TCGA_DU_6408_19860521__slice_22）中，两模型的 Recall 均为 0.7652，但 M4-P 的 FP 像素由 192 增至 877，IoU 由 0.4171 降至 0.1590。两个案例呈现出不同方向的切片级变化。' | Out-Null

    Delete-Paragraph '（5）综合误差特征与改进方向'
    Delete-Paragraph '综合三个训练随机种子，M4-P 的 Positive Macro IoU 为 0.7664±0.0086，Positive Macro Dice 为 0.8425±0.0101，Micro Precision 和 Micro Recall 分别为 0.8710±0.0058 和 0.8983±0.0239；空切片误报率为 18.18%±2.22 个百分点。Recall 整体高于 Precision，结合过分割和空切片误报案例，说明当前模型在提高病灶覆盖率的同时仍存在背景区域误激活。另一方面，小病灶组较低的 IoU、三张完全漏检切片以及欠分割案例表明，模型对微小、低对比度或边缘层面的病灶仍不稳定。'
    Delete-Paragraph '根据上述误差模式，后续可从四个方面改进：第一，在训练阶段增加小病灶重采样、尺度敏感损失或困难样本挖掘，提高微小前景的梯度贡献；第二，引入连通域筛选、最小面积约束或基于验证集的阈值校准，抑制离散假阳性区域，但应同时评估是否导致小病灶被删除；第三，利用相邻切片或三维上下文约束预测的层间连续性，降低病灶起止层面的漏检；第四，按患者、病灶面积、图像质量和位置进行分层评价，并由影像科专家对边界合理性和临床可接受性进行盲法复核。当前结果仅说明模型在本数据集和固定测试清单上的典型行为，不构成外部泛化或临床可用性的证据。'

    # 第五部分：集中呈现综合结果解释、机制假设、不确定性和应用边界。
    Replace-SectionBody '5.1 整体分割表现' '5.2 不同组件的作用' @(
        '在固定的患者级测试集和三个训练随机种子下，M4-P 的 Positive Macro IoU、Positive Dice、Micro IoU、Micro Precision 和 Micro Recall 分别为 0.7664±0.0086、0.8425±0.0101、0.7926±0.0153、0.8710±0.0058 和 0.8983±0.0239。其中，Positive Macro IoU 和 Positive Dice 的均值均为八种配置中最高，说明完整方案在以阳性切片等权重叠质量为主要目标时取得了当前实验中的最佳总体结果。由于各配置复用同一患者划分并采用相同训练协议，这一比较主要反映模型配置差异和训练随机性，而不是数据划分变化。',
        '与 E0 相比，M4-P 的 Positive Macro IoU 严格配对提升为 0.1129±0.0249，三个随机种子的差值均为正；患者级平均差值为 0.1053，bootstrap 95% 区间为[0.0181, 0.2335]。从报告均值作描述性比较，M4-P 的 Positive Dice 和 Micro IoU 分别提高 0.1083 和 0.1106，Micro Recall 提高 0.1530，而 Micro Precision 降低 0.0182。因而，完整方案的优势主要体现为病灶区域的重叠度和覆盖率提高，并非所有评价维度同步改善。',
        '值得注意的是，完整方案得分最高不等同于各组件均有明确的独立贡献。M4-NP、M4-P−A 和 M4-P−B 的 Positive Macro IoU 分别为 0.7597±0.0013、0.7622±0.0282 和 0.7567±0.0144，与 M4-P 的均值差距均小于 0.01；部分差距与跨随机种子波动处于同一数量级。因此，模型排名适合描述当前配置的总体表现，却不足以单独回答“哪一个组件有效”，后者必须依据第 4.6 节预先定义的严格配对关系判断。',
        '本研究的结果与既往采用更强编码器或残差结构改进 U-Net 的技术路线在方向上相符[5][6]，但不能进行数值上的直接横向比较。既往工作使用的数据集、肿瘤区域定义、输入序列、二维或三维建模方式及评价单位并不完全一致；本文只处理当前清单中的二维切片级二值病灶掩膜。将本文分数与 BraTS 多区域分割或其他病种研究的 Dice 直接排序，会混淆任务难度和评价协议差异。',
        '因此，对整体结果较准确的表述是：在当前 kaggle_3m 清单、固定患者级划分、阈值和训练协议下，M4-P 相对 E0 获得了方向一致的总体提升，并在主要 Positive 指标上取得最高均值；这一证据支持把 M4-P 作为后续分析的候选模型，但尚不足以证明其具有跨数据集、跨设备或临床环境中的普遍优势。'
    )

    Replace-SectionBody '5.2 不同组件的作用' '5.3 指标之间的权衡' @(
        '组件消融采用条件配对而不是将八种模型的最终均值任意相减。A、B 和 D 分别在完整方案附近移除三通道输入、轻量增强和 ImageNet 预训练，C 则比较具有相同三通道输入与增强设置的 M4-NP 和 M0-AB。该设计能够减少数据划分和训练协议差异，但每个差值仍是“在其余组件固定条件下”的局部效应，不能自动外推到所有模型组合。',
        'ResNet34 结构 C 的 Positive Macro IoU 配对提升为 0.0637±0.0579，三个随机种子的差值均为正；患者级平均差值为 0.0673，bootstrap 95% 区间为[0.0331, 0.1071]。这一结果在四项组件对照中方向最稳定，并与 ResNet34 系列较早达到较高验证指标的曲线现象相互印证。残差连接和更深的特征层级可能有利于提取病灶的多尺度表征，这与相关残差 U-Net 研究的设计动机一致[5]。不过，当前对照同时改变了编码器深度、参数化方式和特征金字塔，故结论应限定为“ResNet34 编码器配置的贡献”，不宜进一步归因于某一个内部结构细节。',
        '三通道输入 A 的配对提升为 0.0041±0.0366，患者级平均差值为 0.0027，bootstrap 95% 区间为[−0.0112, 0.0221]。其均值接近于零，且训练种子或患者层面的方向并不一致。一个可能原因是当前所谓三通道输入并非每名患者都具有三个完整且相互独立的 MRI 模态：缺失序列会由 FLAIR 填充，通道间可能存在信息冗余，模型也可能学习到与缺失模式相关的非病灶特征。既往研究已指出，多模态 MRI 的缺失信息需要被显式处理[13]。因此，本实验不否认多模态信息的潜在价值，而是说明当前通道构造和样本规模尚未给出稳定的独立增益证据。',
        '轻量增强 B 的配对提升为 0.0096±0.0148，患者级平均差值为 0.0112，但 bootstrap 95% 区间为[−0.0091, 0.0357]。增强可能通过扩大姿态和强度扰动范围降低模型对训练样本的记忆，但其收益取决于变换幅度是否符合 MRI 成像与解剖结构。当前仅验证了一组固定的轻量增强策略，不能据此判断增强总体无效，也不能把均值为正写成确定性改善。后续应分别考察几何变换、强度变换及其组合，并报告它们对小病灶保留和边界误差的影响。',
        'ImageNet 预训练 D 的配对提升为 0.0066±0.0075，患者级平均差值为 0.0001，bootstrap 95% 区间为[−0.0224, 0.0258]。自然图像预训练可能改善优化初期的特征初始化，却不保证在灰度 MRI 域中形成稳定的最终性能增益。M4-P 与 M4-NP 的接近结果表明，在当前数据量和训练设置下，ResNet34 结构本身与 ImageNet 初始化的作用需要区分；预训练是否受益还可能受到输入通道映射、微调学习率和冻结策略影响。',
        '完整方案相对 E0 的提升大于任一单组件配对差值，但这不能被解释为四个组件贡献的简单相加。各对照的参照配置不同，组件之间可能存在互补、冗余或抑制关系，而三随机种子不足以稳定估计高阶交互效应。例如，多通道输入在普通 U-Net 与 ResNet34 编码器中的作用可能不同，预训练的效果也可能依赖增强强度。若要进一步识别交互作用，需要采用更完整的因子设计并增加训练重复次数。',
        '综合而言，当前证据最支持 ResNet34 编码器配置是相对稳定的改进因素；三通道输入、轻量增强和 ImageNet 预训练仅表现为较小且不稳定的条件性均值变化。论文结论应区分“完整方案有效”“某组件具有稳定贡献”和“某组件可能在特定条件下有效”这三个层次，避免由最高分配置反推所有组成部分均不可或缺。'
    )

    Replace-SectionBody '5.3 指标之间的权衡' '5.4 患者级不确定性' @(
        '本研究同时报告 Positive Macro 指标、Micro 指标和空切片误报率，是因为它们回答的问题不同。Positive Macro IoU 和 Dice 对每张阳性切片等权，能够突出病灶重叠质量，但不评价真实空切片上的误报；Micro 指标先累积所有像素，更容易受到大病灶和大量背景像素影响；空切片误报率则直接反映模型是否在无病灶切片上错误激活。只选择其中一个分数，可能掩盖具有实际意义的错误模式。',
        'M4-P 的 Micro Recall 为 0.8983±0.0239，高于 E0 的 0.7453±0.0152；但其 Micro Precision 为 0.8710±0.0058，低于 E0 的 0.8892±0.0295，空切片误报率也由 E0 的 15.81%±6.20% 变为 18.18%±2.22%。这些均值差异表明，完整方案倾向于覆盖更多病灶像素，同时付出一定背景误激活代价。由于表中 Precision、Recall 和误报率未逐项给出患者级置信区间，这里仅作描述性权衡，不将均值变化解释为统计学上的确定差异。',
        '图 4-2 的过分割和欠分割案例进一步说明相同总体得分可以由不同空间错误构成。过分割样本的 Recall 为 0.7652，但 Precision 仅为 0.1671，表现为多个离散假阳性区域；欠分割样本的 Precision 达 0.9908，但 Recall 仅为 0.3106，大部分真实病灶未被覆盖。对病灶定量分析而言，前者可能高估病灶范围，后者可能低估病灶范围，因此 Precision 或 Recall 的单侧优势都不能替代对轮廓和混淆像素的联合检查。',
        '小病灶是当前模型较明确的薄弱场景。在种子 42 下，最小病灶面积组的平均 IoU 为 0.5782，低于其余三组的 0.7784、0.8834 和 0.8593，且 3 张完全漏检切片全部落在最小面积组。可能的解释包括前景像素在损失中的贡献有限、多次下采样削弱细小结构，以及固定 0.5 阈值对低置信度病灶不够敏感。不过，该分析以切片为单位，同一患者的切片并非独立样本，且未针对病灶面积进行预设假设检验，故只能视为误差定位线索，不能据此断言病灶面积与模型失败存在因果关系。',
        '空切片误报则揭示另一侧风险。种子 42 的 352 张空切片中有 55 张出现至少一个假阳性像素，其中示例切片被错误预测了 4584 个前景像素，约占整幅图像的 7.0%。虽然空切片平均预测前景比例只有 0.0906%，少量严重误报仍可能增加人工复核负担，并在按切片累计面积或重建体积时造成偏差。因此，较高的总体 Accuracy 或 Specificity 不能被用来证明背景误检已经得到解决。',
        '针对上述两类相反错误，改进策略需要联合验证而非单向优化。训练阶段可考察小病灶重采样、困难样本挖掘、边界或尺度敏感损失，以增加微小前景的学习信号；推理阶段可在验证集上校准阈值，并评估连通域筛选和最小面积约束。后处理虽然可能删除离散假阳性，也可能同时移除真正的小病灶，因此应绘制不同阈值下的 Precision–Recall 与空切片误报变化，而不能只报告处理后的最高 IoU。',
        '此外，二维切片模型缺乏相邻层面的连续性约束。利用相邻切片、2.5D 输入或三维模型可能减少病灶起止层面的漏检和孤立误报，但也会改变计算量、数据组织和评价单位。后续实验应按患者、病灶面积、图像质量和解剖位置进行分层报告，并由影像科专家盲法评价边界是否可接受，使像素级改进与实际复核成本建立更清晰的联系[10]。'
    )

    Replace-SectionBody '5.4 患者级不确定性' '5.5 局限性与伦理说明' @(
        '三个训练随机种子用于观察固定数据划分下由参数初始化、批次顺序和随机增强带来的优化波动。曲线中，M4-P 的最佳验证 Positive Macro IoU 为 0.7547±0.0058，最佳 epoch 为 25、25 和 34；E2-B 的最佳 epoch 则为 48、24 和 111，跨种子收敛位置差异更大。该结果说明不同配置对训练随机性的敏感程度不同，也说明仅报告单次最佳运行可能夸大或低估模型表现。',
        'E0 的最佳训练—验证指标平均差距为 0.3120，明显高于 M4-P 的 0.0573；同时，多数配置的验证损失在训练中后期趋于平台或波动，而训练损失继续下降。这些现象与 E0 更明显的训练—验证偏离一致，因此采用验证指标选取检查点和早停是必要的。不过，训练—验证差距还会受到模型容量、优化速度、增强方式和验证集规模影响，不能仅凭曲线把差距完全归因于过拟合，也不能把较小差距直接等同于更好的测试泛化。',
        '患者级 bootstrap 关注的是固定测试患者之间的差异，与训练随机种子所反映的不确定性并不相同。完整方案相对 E0 的患者级平均差值为 0.1053，95% 区间为[0.0181, 0.2335]；ResNet34 结构对照的区间也保持为正。相比之下，A、B 和 D 的患者级区间均跨越 0。该结果为组件结论提供了患者层面的补充，但测试集只有 10 名患者，区间端点很容易受个别病例影响，bootstrap 在此主要用于描述而不是形成广泛的总体推断。',
        '需要特别区分两类“重复”：三个随机种子是在同一组患者上重复训练，并没有带来新的独立患者；525 张测试切片也不能被视为 525 个相互独立的临床样本，因为同一患者的相邻切片高度相关。若把切片数直接当作样本量，会人为缩小不确定性。本文采用患者级划分和患者级配对汇总，能够控制明显的跨集合泄漏风险[8]，但无法弥补测试患者数量有限的问题。',
        '图 4-4 同样显示平均提升具有病例依赖性：M4-P 在一个切片上将 IoU 从 0.1259 提高至 0.9658，却在另一个切片上因假阳性增加使 IoU 从 0.4171 降至 0.1590。总体均值为正不意味着每名患者、每个病灶或每张切片都会受益。后续应报告患者级差值分布和失败患者特征，并预先定义基于病灶大小、成像质量和缺失通道模式的亚组分析，避免只展示有利案例。',
        '外部泛化仍是最重要的未决问题。既往研究表明，在标准化挑战数据上训练的模型迁移至另一机构临床 MRI 时性能可能下降[7]。本文尚未覆盖不同医院、扫描设备、层厚和预处理流程，也未开展时间外验证。较合理的下一步是冻结当前模型与阈值，在独立来源数据上进行患者级测试，并在任何再训练或域适配之前单独报告原始外部性能。'
    )

    Replace-SectionBody '5.5 局限性与伦理说明' '6 结论与未来工作' @(
        '第一，数据来自单一公开来源，当前清单共 104 名患者，测试集只有 10 名患者，且患者标签均为 LGG。有限的患者数量和单一病种范围限制了统计稳定性与适用人群，结果不能外推至其他肿瘤类型、不同疾病阶段或真实医院人群。',
        '第二，本文任务是二维切片级二值分割，只区分病灶与背景，不进行肿瘤亚区或肿瘤类型的像素级多分类，也未重建和评价三维病灶体积。二维模型忽略相邻切片上下文，因此本文指标不能直接代表三维体积测量精度或纵向随访中的体积变化可靠性。',
        '第三，三通道数据并非所有患者均具有完整且独立的 MRI 序列，缺失通道由 FLAIR 填充。该处理保证了输入维度一致，却可能引入通道冗余和缺失模式偏差。当前消融只能评价这一具体输入构造，不能概括真正完整的多模态 MRI 融合效果。',
        '第四，研究仅完成内部固定测试，尚无外部医院、跨设备或时间外验证，也未与放射科医生的重复勾画差异进行比较。已有研究提示跨机构数据差异会削弱模型表现[7]，专家中心评价也能补充重叠指标无法表达的边界可接受性[10]。因此，当前结果不构成临床有效性证据。',
        '第五，模型使用固定阈值 0.5，尚未系统评价概率校准、阈值敏感性、预测不确定性和后处理规则。误差分析主要基于 M4-P 的种子 42 运行及客观筛选的典型样本，能够展示失败模式，但不能替代全部随机种子和全部患者的系统分层统计。',
        '第六，三次训练重复只能初步描述随机优化波动，患者级 bootstrap 又受 10 名测试患者限制。本文未对多项指标和多个组件执行正式的多重比较推断，因此应以效应大小、方向一致性和区间宽度为重点，不使用“显著优于”或“证明有效”等超出当前设计的措辞。',
        '从伦理和使用边界看，本研究使用公开数据开展方法学验证，模型只能作为研究性辅助分析工具。任何面向真实患者的应用都应完成数据授权与隐私审查、外部验证、亚组公平性分析、校准、失败告警和临床专家复核；模型输出不得替代影像科医生诊断、治疗决策或人工勾画。尤其对于完全漏检和大面积空切片误报，应在临床流程设计中保留明确的人工复核与回退机制。',
        '综上，第五部分的结论边界与第四部分的实验事实保持一致：当前证据支持 M4-P 在固定内部测试中的总体改善，并较明确地指向 ResNet34 编码器配置；其余组件、患者亚组表现以及临床泛化仍需通过更大样本、更多训练重复和独立外部验证进一步确认。'
    )

    # 写回 document.xml。
    $entry.Delete()
    $newEntry = $archive.CreateEntry('word/document.xml', [System.IO.Compression.CompressionLevel]::Optimal)
    $stream = $newEntry.Open()
    $settings = [System.Xml.XmlWriterSettings]::new()
    $settings.Encoding = [System.Text.UTF8Encoding]::new($false)
    $settings.Indent = $false
    $settings.OmitXmlDeclaration = $false
    $writer = [System.Xml.XmlWriter]::Create($stream, $settings)
    try { $xml.Save($writer) } finally { $writer.Dispose(); $stream.Dispose() }
}
finally {
    $archive.Dispose()
}

Write-Output $outputPath
