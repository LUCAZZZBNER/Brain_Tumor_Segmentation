param(
    [Parameter(Mandatory = $true)]
    [string]$Output
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$outputPath = [IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
$outputDirectory = [IO.Path]::GetDirectoryName($outputPath)
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}

$canvasWidth = 2600
$canvasHeight = 2660
$dpi = 600
$bitmap = [Drawing.Bitmap]::new($canvasWidth, $canvasHeight)
$bitmap.SetResolution($dpi, $dpi)
$graphics = [Drawing.Graphics]::FromImage($bitmap)

try {
    $graphics.SmoothingMode = [Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.PixelOffsetMode = [Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $graphics.TextRenderingHint = [Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.Clear([Drawing.Color]::White)

    $navy = [Drawing.Color]::FromArgb(24, 58, 91)
    $border = [Drawing.Color]::FromArgb(61, 84, 105)
    $encoderFill = [Drawing.Color]::FromArgb(224, 239, 250)
    $decoderFill = [Drawing.Color]::FromArgb(226, 244, 230)
    $bottleneckFill = [Drawing.Color]::FromArgb(248, 228, 233)
    $ioFill = [Drawing.Color]::FromArgb(252, 246, 220)
    $skip = [Drawing.Color]::FromArgb(221, 103, 32)
    $muted = [Drawing.Color]::FromArgb(82, 92, 103)

    $navyBrush = [Drawing.SolidBrush]::new($navy)
    $skipBrush = [Drawing.SolidBrush]::new($skip)
    $mutedBrush = [Drawing.SolidBrush]::new($muted)
    $borderPen = [Drawing.Pen]::new($border, 7)
    $forwardPen = [Drawing.Pen]::new($navy, 8)
    $forwardPen.CustomEndCap = [Drawing.Drawing2D.AdjustableArrowCap]::new(3.2, 4.2, $true)
    $skipPen = [Drawing.Pen]::new($skip, 7)
    $skipPen.DashStyle = [Drawing.Drawing2D.DashStyle]::Dash
    $skipPen.CustomEndCap = [Drawing.Drawing2D.AdjustableArrowCap]::new(3.0, 4.0, $true)

    $titleFont = [Drawing.Font]::new('Arial', 14.0, [Drawing.FontStyle]::Bold)
    $subtitleFont = [Drawing.Font]::new('Arial', 8.5, [Drawing.FontStyle]::Regular)
    $boxTitleFont = [Drawing.Font]::new('Arial', 9.8, [Drawing.FontStyle]::Bold)
    $boxTextFont = [Drawing.Font]::new('Arial', 8.2, [Drawing.FontStyle]::Regular)
    $footerFont = [Drawing.Font]::new('Arial', 7.4, [Drawing.FontStyle]::Regular)
    $footerBoldFont = [Drawing.Font]::new('Arial', 7.4, [Drawing.FontStyle]::Bold)

    $center = [Drawing.StringFormat]::new()
    $center.Alignment = [Drawing.StringAlignment]::Center
    $center.LineAlignment = [Drawing.StringAlignment]::Center

    function New-RoundedRectanglePath {
        param([float]$X, [float]$Y, [float]$Width, [float]$Height, [float]$Radius)
        $path = [Drawing.Drawing2D.GraphicsPath]::new()
        $diameter = 2 * $Radius
        $path.AddArc($X, $Y, $diameter, $diameter, 180, 90)
        $path.AddArc($X + $Width - $diameter, $Y, $diameter, $diameter, 270, 90)
        $path.AddArc($X + $Width - $diameter, $Y + $Height - $diameter, $diameter, $diameter, 0, 90)
        $path.AddArc($X, $Y + $Height - $diameter, $diameter, $diameter, 90, 90)
        $path.CloseFigure()
        return $path
    }

    function Draw-Box {
        param(
            [int]$X,
            [int]$Y,
            [Drawing.Color]$Fill,
            [string]$Title,
            [string]$Line1,
            [string]$Line2
        )
        $path = New-RoundedRectanglePath $X $Y $boxWidth $boxHeight 24
        $fillBrush = [Drawing.SolidBrush]::new($Fill)
        try {
            $graphics.FillPath($fillBrush, $path)
            $graphics.DrawPath($borderPen, $path)
        }
        finally {
            $fillBrush.Dispose()
            $path.Dispose()
        }
        $graphics.DrawString($Title, $boxTitleFont, $navyBrush, [Drawing.RectangleF]::new($X + 24, $Y + 12, $boxWidth - 48, 85), $center)
        $graphics.DrawString($Line1, $boxTextFont, $navyBrush, [Drawing.RectangleF]::new($X + 24, $Y + 80, $boxWidth - 48, 82), $center)
        $graphics.DrawString($Line2, $boxTextFont, $navyBrush, [Drawing.RectangleF]::new($X + 24, $Y + 138, $boxWidth - 48, 82), $center)
    }

    function Draw-VerticalArrow {
        param([int]$X, [int]$FromY, [int]$ToY)
        $graphics.DrawLine($forwardPen, $X, $FromY, $X, $ToY)
    }

    $graphics.DrawString('M4-P: ResNet34 U-Net', $titleFont, $navyBrush, [Drawing.RectangleF]::new(100, 10, 2400, 150), $center)
    # $graphics.DrawString('Code-faithful 2D segmentation architecture', $subtitleFont, $mutedBrush, [Drawing.RectangleF]::new(100, 138, 2400, 55), $center)
    $graphics.DrawString('Dashed orange arrows: concatenated skip features', $subtitleFont, $skipBrush, [Drawing.RectangleF]::new(100, 193, 2400, 70), $center)

    $boxWidth = 1000
    $boxHeight = 230
    $leftX = 100
    $rightX = 1500
    $centerX = 800
    $leftMid = $leftX + [int]($boxWidth / 2)
    $rightMid = $rightX + [int]($boxWidth / 2)

    $y0 = 310
    $y1 = 670
    $y2 = 1030
    $y3 = 1390
    $y4 = 1750
    $y5 = 2200

    Draw-Box $leftX  $y0 $ioFill          'Input'                    '3 channels'                         '3 x 256 x 256'
    Draw-Box $leftX  $y1 $encoderFill     'Stem'                     '7 x 7 Conv s2; BN; ReLU'            '64 x 128 x 128'
    Draw-Box $leftX  $y2 $encoderFill     'MaxPool + Layer 1'        '3 BasicBlocks'                     '64 x 64 x 64'
    Draw-Box $leftX  $y3 $encoderFill     'ResNet34 Layer 2'         '4 BasicBlocks'                     '128 x 32 x 32'
    Draw-Box $leftX  $y4 $encoderFill     'ResNet34 Layer 3'         '6 BasicBlocks'                     '256 x 16 x 16'
    Draw-Box $centerX $y5 $bottleneckFill 'Layer 4 / bottleneck'     '3 BasicBlocks'                     '512 x 8 x 8'

    Draw-Box $rightX $y4 $decoderFill     'Decoder 1'                'TConv 512 -> 256 + L3'             '256 x 16 x 16'
    Draw-Box $rightX $y3 $decoderFill     'Decoder 2'                'TConv 256 -> 128 + L2'             '128 x 32 x 32'
    Draw-Box $rightX $y2 $decoderFill     'Decoder 3'                'TConv 128 -> 64 + L1'              '64 x 64 x 64'
    Draw-Box $rightX $y1 $decoderFill     'Decoder 4'                'TConv 64 -> 32 + Stem'             '32 x 128 x 128'
    Draw-Box $rightX $y0 $ioFill          'Decoder 5 (no skip)'      'TConv 32->16; 1x1 head'            'Logits: 1 x 256 x 256'

    Draw-VerticalArrow $leftMid ($y0 + $boxHeight + 12) ($y1 - 14)
    Draw-VerticalArrow $leftMid ($y1 + $boxHeight + 12) ($y2 - 14)
    Draw-VerticalArrow $leftMid ($y2 + $boxHeight + 12) ($y3 - 14)
    Draw-VerticalArrow $leftMid ($y3 + $boxHeight + 12) ($y4 - 14)

    $graphics.DrawLine($forwardPen, $leftMid, $y4 + $boxHeight + 12, $leftMid, $y5 - 65)
    $graphics.DrawLine($forwardPen, $leftMid, $y5 - 65, 1300, $y5 - 65)
    $graphics.DrawLine($forwardPen, 1300, $y5 - 65, 1300, $y5 - 14)

    $graphics.DrawLine($forwardPen, $centerX + $boxWidth + 12, $y5 + [int]($boxHeight / 2), $rightMid, $y5 + [int]($boxHeight / 2))
    $graphics.DrawLine($forwardPen, $rightMid, $y5 + [int]($boxHeight / 2), $rightMid, $y4 + $boxHeight + 14)

    Draw-VerticalArrow $rightMid ($y4 - 12) ($y3 + $boxHeight + 14)
    Draw-VerticalArrow $rightMid ($y3 - 12) ($y2 + $boxHeight + 14)
    Draw-VerticalArrow $rightMid ($y2 - 12) ($y1 + $boxHeight + 14)
    Draw-VerticalArrow $rightMid ($y1 - 12) ($y0 + $boxHeight + 14)

    foreach ($rowY in @($y1, $y2, $y3, $y4)) {
        $midY = $rowY + [int]($boxHeight / 2)
        $graphics.DrawLine($skipPen, $leftX + $boxWidth + 12, $midY, $rightX - 14, $midY)
    }

    $graphics.DrawString(
        'Encoder: ImageNet pretrained. Decoder: TConv + DoubleConv.',
        $footerFont,
        $mutedBrush,
        [Drawing.RectangleF]::new(160, 2465, 2280, 55),
        $center
    )
    $graphics.DrawString(
        'No decoder BN or dropout.',
        $footerBoldFont,
        $mutedBrush,
        [Drawing.RectangleF]::new(160, 2520, 2280, 62),
        $center
    )
    $graphics.DrawString(
        'DoubleConv = [3 x 3 Conv + ReLU] x 2. Output: raw logits.',
        $footerBoldFont,
        $mutedBrush,
        [Drawing.RectangleF]::new(160, 2575, 2280, 62),
        $center
    )

    $bitmap.Save($outputPath, [Drawing.Imaging.ImageFormat]::Png)
}
finally {
    foreach ($object in @(
        $footerBoldFont, $footerFont, $boxTextFont, $boxTitleFont, $subtitleFont, $titleFont,
        $skipPen, $forwardPen, $borderPen, $mutedBrush, $skipBrush, $navyBrush, $center
    )) {
        if ($null -ne $object) { $object.Dispose() }
    }
    $graphics.Dispose()
    $bitmap.Dispose()
}

Get-Item -LiteralPath $outputPath | Select-Object FullName, Length
