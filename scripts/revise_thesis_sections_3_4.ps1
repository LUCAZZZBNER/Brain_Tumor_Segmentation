param(
    [string]$InputDocx = "reports\论文中文草稿框架_摘要修改版.docx",
    [string]$OutputDocx = "reports\论文中文草稿框架_第三四部分修改版.docx",
    [string]$FlowchartPng = "tmp\thesis_revision\section3_workflow.png"
)

$ErrorActionPreference = "Stop"

function Get-AbsolutePath {
    param([string]$Path, [switch]$AllowMissing)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    $combined = Join-Path (Get-Location) $Path
    if ($AllowMissing) {
        return [System.IO.Path]::GetFullPath($combined)
    }
    return (Resolve-Path -LiteralPath $combined).Path
}

function New-RoundedRectanglePath {
    param(
        [float]$X,
        [float]$Y,
        [float]$Width,
        [float]$Height,
        [float]$Radius
    )
    $path = [System.Drawing.Drawing2D.GraphicsPath]::new()
    $diameter = 2 * $Radius
    $path.AddArc($X, $Y, $diameter, $diameter, 180, 90)
    $path.AddArc($X + $Width - $diameter, $Y, $diameter, $diameter, 270, 90)
    $path.AddArc($X + $Width - $diameter, $Y + $Height - $diameter, $diameter, $diameter, 0, 90)
    $path.AddArc($X, $Y + $Height - $diameter, $diameter, $diameter, 90, 90)
    $path.CloseFigure()
    return $path
}

function New-WorkflowFigure {
    param([string]$Path)

    Add-Type -AssemblyName System.Drawing
    $directory = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $directory)) {
        New-Item -ItemType Directory -Path $directory | Out-Null
    }

    $steps = @(
        "数据清单构建与哈希校验",
        "患者级训练集、验证集与测试集划分",
        "通道选择、尺寸统一、归一化与掩膜二值化",
        "训练集同步增强与正/空掩膜采样",
        "U-Net 或 ResNet34-U-Net 模型训练",
        "验证集选择最佳 checkpoint 与二值化阈值",
        "冻结配置后在测试集进行独立评估",
        "指标汇总、患者级配对分析与结果可视化"
    )

    $width = 1800
    $height = 1420
    $bitmap = [System.Drawing.Bitmap]::new($width, $height)
    $bitmap.SetResolution(300, 300)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    $graphics.Clear([System.Drawing.Color]::White)

    $mainFont = [System.Drawing.Font]::new("Microsoft YaHei", 38, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
    $numberFont = [System.Drawing.Font]::new("Microsoft YaHei", 34, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    $textBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(31, 51, 73))
    $numberBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::White)
    $linePen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(45, 91, 132), 6)
    $linePen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $linePen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
    $fills = @(
        [System.Drawing.Color]::FromArgb(231, 242, 250),
        [System.Drawing.Color]::FromArgb(239, 246, 252)
    )
    $borderPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(45, 91, 132), 4)
    $numberFill = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(45, 91, 132))
    $stringFormat = [System.Drawing.StringFormat]::new()
    $stringFormat.Alignment = [System.Drawing.StringAlignment]::Center
    $stringFormat.LineAlignment = [System.Drawing.StringAlignment]::Center

    $boxX = 150
    $boxWidth = 1500
    $boxHeight = 118
    $top = 50
    $gap = 58
    $circleSize = 70

    try {
        for ($index = 0; $index -lt $steps.Count; $index++) {
            $boxY = $top + $index * ($boxHeight + $gap)
            $pathObject = New-RoundedRectanglePath -X $boxX -Y $boxY -Width $boxWidth -Height $boxHeight -Radius 25
            $fillBrush = [System.Drawing.SolidBrush]::new($fills[$index % 2])
            try {
                $graphics.FillPath($fillBrush, $pathObject)
                $graphics.DrawPath($borderPen, $pathObject)
            }
            finally {
                $fillBrush.Dispose()
                $pathObject.Dispose()
            }

            $circleX = $boxX + 34
            $circleY = $boxY + ($boxHeight - $circleSize) / 2
            $graphics.FillEllipse($numberFill, $circleX, $circleY, $circleSize, $circleSize)
            $graphics.DrawString(($index + 1).ToString(), $numberFont, $numberBrush, [System.Drawing.RectangleF]::new($circleX, $circleY, $circleSize, $circleSize), $stringFormat)

            $textRect = [System.Drawing.RectangleF]::new($boxX + 135, $boxY, $boxWidth - 200, $boxHeight)
            $graphics.DrawString($steps[$index], $mainFont, $textBrush, $textRect, $stringFormat)

            if ($index -lt $steps.Count - 1) {
                $arrowX = $width / 2
                $startY = $boxY + $boxHeight + 8
                $endY = $boxY + $boxHeight + $gap - 16
                $graphics.DrawLine($linePen, $arrowX, $startY, $arrowX, $endY)
                $arrow = @(
                    [System.Drawing.PointF]::new($arrowX - 17, $endY - 10),
                    [System.Drawing.PointF]::new($arrowX + 17, $endY - 10),
                    [System.Drawing.PointF]::new($arrowX, $endY + 14)
                )
                $graphics.FillPolygon($numberFill, $arrow)
            }
        }
        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        $stringFormat.Dispose()
        $numberFill.Dispose()
        $borderPen.Dispose()
        $linePen.Dispose()
        $numberBrush.Dispose()
        $textBrush.Dispose()
        $numberFont.Dispose()
        $mainFont.Dispose()
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

function Find-TextRange {
    param(
        $Document,
        [string]$Text
    )
    $range = $Document.Content.Duplicate
    $find = $range.Find
    $find.ClearFormatting()
    $find.Text = $Text
    $find.Forward = $true
    $find.Wrap = 0
    $find.Format = $false
    if (-not $find.Execute()) {
        throw "未在文档中找到文本：$Text"
    }
    return $range
}

function Set-ChineseFont {
    param($Range, [string]$FontName = "宋体", [double]$Size = 10.5)
    $Range.Font.Name = $FontName
    $Range.Font.NameFarEast = $FontName
    $Range.Font.Size = $Size
}

$inputPath = Get-AbsolutePath -Path $InputDocx
$outputPath = Get-AbsolutePath -Path $OutputDocx -AllowMissing
$figurePath = Get-AbsolutePath -Path $FlowchartPng -AllowMissing

if (Test-Path -LiteralPath $outputPath) {
    throw "输出文件已存在，为避免覆盖已停止：$outputPath"
}

$outputDirectory = Split-Path -Parent $outputPath
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}

New-WorkflowFigure -Path $figurePath
Copy-Item -LiteralPath $inputPath -Destination $outputPath

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($outputPath, $false, $false)

    # 1. 将 3.1 中的文本占位流程替换为正式流程图和图注。
    $heading31 = Find-TextRange -Document $document -Text "3.1 总体流程"
    $heading32 = Find-TextRange -Document $document -Text "3.2 数据集与清洁清单"
    $body31 = $document.Range($heading31.Paragraphs.Item(1).Range.End, $heading32.Paragraphs.Item(1).Range.Start)
    $body31.Delete()

    $heading31 = Find-TextRange -Document $document -Text "3.1 总体流程"
    $insertPosition = $heading31.Paragraphs.Item(1).Range.End
    $introText = "本文总体流程如图 3-1 所示，依次完成数据核验与患者级划分、预处理与训练、模型选择及测试评估，并在冻结实验配置后进行指标汇总、患者级配对分析和结果可视化。"
    $introRange = $document.Range($insertPosition, $insertPosition)
    $introRange.Text = $introText + "`r"
    $introParagraph = $document.Range($insertPosition, $insertPosition + $introText.Length).Paragraphs.Item(1)
    $introParagraph.Range.Style = $document.Styles.Item(-1)
    $introParagraph.Alignment = 3
    $introParagraph.SpaceAfter = 6
    Set-ChineseFont -Range $introParagraph.Range

    $picturePosition = $insertPosition + $introText.Length + 1
    $pictureRange = $document.Range($picturePosition, $picturePosition)
    $picture = $document.InlineShapes.AddPicture($figurePath, $false, $true, $pictureRange)
    $picture.LockAspectRatio = -1
    $picture.Width = $word.InchesToPoints(5.8)
    $picture.AlternativeText = "本文研究总体流程：数据核验、患者级划分、预处理、训练、验证选择、测试评估和结果分析。"
    $pictureParagraph = $picture.Range.Paragraphs.Item(1)
    $pictureParagraph.Alignment = 1
    $pictureParagraph.SpaceBefore = 3
    $pictureParagraph.SpaceAfter = 3
    $pictureParagraph.KeepWithNext = -1

    $captionText = "图 3-1 本文研究的总体流程"
    $captionPosition = $picture.Range.End
    $captionRange = $document.Range($captionPosition, $captionPosition)
    $captionRange.Text = "`r" + $captionText + "`r"
    $captionFound = Find-TextRange -Document $document -Text $captionText
    $captionParagraph = $captionFound.Paragraphs.Item(1)
    try { $captionParagraph.Range.Style = $document.Styles.Item(-35) } catch { $captionParagraph.Range.Style = $document.Styles.Item(-1) }
    $captionParagraph.Alignment = 1
    $captionParagraph.SpaceBefore = 0
    $captionParagraph.SpaceAfter = 6
    $captionParagraph.KeepWithNext = -1
    Set-ChineseFont -Range $captionParagraph.Range -Size 10.5

    # 2. 把 3.5 的模型配置表移出方法章节，保留模型结构说明。
    $heading35 = Find-TextRange -Document $document -Text "3.5 模型与实验因素"
    $heading36 = Find-TextRange -Document $document -Text "3.6 损失函数与训练策略"
    $configurationTable = $null
    foreach ($table in $document.Tables) {
        if ($table.Range.Start -gt $heading35.Paragraphs.Item(1).Range.Start -and $table.Range.Start -lt $heading36.Paragraphs.Item(1).Range.Start) {
            $configurationTable = $table
            break
        }
    }
    if ($null -eq $configurationTable) {
        throw "未找到 3.5 节中的模型配置表。"
    }
    $configurationTableXml = $configurationTable.Range.WordOpenXML

    $body35 = $document.Range($heading35.Paragraphs.Item(1).Range.End, $heading36.Paragraphs.Item(1).Range.Start)
    $body35.Delete()
    $heading35 = Find-TextRange -Document $document -Text "3.5 模型与实验因素"
    $heading35.Text = "3.5 模型结构"

    $heading35 = Find-TextRange -Document $document -Text "3.5 模型结构"
    $modelDescription1 = "本文比较普通 U-Net 与 ResNet34-U-Net 两类二维编码器—解码器模型。普通 U-Net 采用双卷积模块逐级提取特征；ResNet34-U-Net 使用 ResNet34 形成编码特征金字塔，并通过跳跃连接将不同尺度的编码特征传递给解码器。"
    $modelDescription2 = "两类模型均输出单通道病灶概率图，可接收单通道 FLAIR 或三通道输入。为区分输入、增强、编码器结构和预训练的影响，具体模型配置与对照关系统一放在第 4.3 节说明。"
    $modelInsertPosition = $heading35.Paragraphs.Item(1).Range.End
    $modelRange = $document.Range($modelInsertPosition, $modelInsertPosition)
    $modelRange.Text = $modelDescription1 + "`r" + $modelDescription2 + "`r"
    $modelBodyRange = $document.Range($modelInsertPosition, $modelInsertPosition + $modelDescription1.Length + $modelDescription2.Length + 2)
    $modelBodyRange.Style = $document.Styles.Item(-1)
    $modelBodyRange.ParagraphFormat.Alignment = 3
    $modelBodyRange.ParagraphFormat.SpaceAfter = 6
    Set-ChineseFont -Range $modelBodyRange

    # 3. 顺延第四章小节编号，并新增 4.3 模型配置与对照设计。
    $renumbering = @(
        @{ Old = "4.6 可视化与误差分析【部分待补充】"; New = "4.7 可视化与误差分析【部分待补充】" },
        @{ Old = "4.5 组件消融结果（当前已有数据）"; New = "4.6 组件消融结果（当前已有数据）" },
        @{ Old = "4.4 模型对比结果（当前已有数据）"; New = "4.5 模型对比结果（当前已有数据）" },
        @{ Old = "4.3 多随机种子实验"; New = "4.4 多随机种子实验" }
    )
    foreach ($item in $renumbering) {
        $range = Find-TextRange -Document $document -Text $item.Old
        $range.Text = $item.New
    }

    $heading44 = Find-TextRange -Document $document -Text "4.4 多随机种子实验"
    $sectionInsertPosition = $heading44.Paragraphs.Item(1).Range.Start
    $sectionHeadingText = "4.3 模型配置与对照设计"
    $sectionLeadText = "为分离三通道输入、轻量数据增强、ResNet34 编码器结构和 ImageNet 预训练的影响，本文设置八种实验配置，并在相同数据清单与训练协议下进行严格配对比较。"
    $sectionRange = $document.Range($sectionInsertPosition, $sectionInsertPosition)
    $sectionRange.Text = $sectionHeadingText + "`r" + $sectionLeadText + "`r"

    $newHeadingRange = $document.Range($sectionInsertPosition, $sectionInsertPosition + $sectionHeadingText.Length)
    $newHeadingRange.Paragraphs.Item(1).Range.Style = $document.Styles.Item(-3)
    $leadStart = $sectionInsertPosition + $sectionHeadingText.Length + 1
    $leadRange = $document.Range($leadStart, $leadStart + $sectionLeadText.Length)
    $leadRange.Paragraphs.Item(1).Range.Style = $document.Styles.Item(-1)
    $leadRange.ParagraphFormat.Alignment = 3
    $leadRange.ParagraphFormat.SpaceAfter = 6
    Set-ChineseFont -Range $leadRange

    $tableInsertPosition = $leadStart + $sectionLeadText.Length + 1
    $tableInsertRange = $document.Range($tableInsertPosition, $tableInsertPosition)
    $tableInsertRange.InsertXML($configurationTableXml)

    $newHeading = Find-TextRange -Document $document -Text $sectionHeadingText
    $nextHeading = Find-TextRange -Document $document -Text "4.4 多随机种子实验"
    $movedTable = $null
    foreach ($table in $document.Tables) {
        if ($table.Range.Start -gt $newHeading.Paragraphs.Item(1).Range.Start -and $table.Range.Start -lt $nextHeading.Paragraphs.Item(1).Range.Start) {
            $movedTable = $table
            break
        }
    }
    if ($null -eq $movedTable) {
        throw "模型配置表移动到 4.3 节后未能重新定位。"
    }
    $movedTable.AllowAutoFit = $true
    $movedTable.Rows.Alignment = 1

    $factorText = "其中，A 表示三通道输入，B 表示轻量增强，C 表示 ResNet34 结构，D 表示 ImageNet 预训练。组件贡献仅依据严格配对结果讨论，不能因完整模型得分最高而将每个组件都解释为确定有效。"
    $factorPosition = $movedTable.Range.End
    $factorRange = $document.Range($factorPosition, $factorPosition)
    $factorRange.Text = $factorText + "`r"
    $factorParagraph = $document.Range($factorPosition, $factorPosition + $factorText.Length).Paragraphs.Item(1)
    $factorParagraph.Range.Style = $document.Styles.Item(-1)
    $factorParagraph.Alignment = 3
    $factorParagraph.SpaceBefore = 6
    $factorParagraph.SpaceAfter = 6
    Set-ChineseFont -Range $factorParagraph.Range

    $document.Repaginate()
    $document.Save()
    $pageCount = $document.ComputeStatistics(2)
    $inlineShapeCount = $document.InlineShapes.Count
    $tableCount = $document.Tables.Count

    $document.Close($false)
    $document = $null
    $word.Quit()
    $word = $null

    Write-Output "OUTPUT=$outputPath"
    Write-Output "FLOWCHART=$figurePath"
    Write-Output "PAGES=$pageCount"
    Write-Output "INLINE_SHAPES=$inlineShapeCount"
    Write-Output "TABLES=$tableCount"
}
finally {
    if ($null -ne $document) {
        try { $document.Close($false) } catch {}
    }
    if ($null -ne $word) {
        try { $word.Quit() } catch {}
    }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
