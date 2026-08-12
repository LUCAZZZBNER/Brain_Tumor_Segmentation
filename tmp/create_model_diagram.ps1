param([Parameter(Mandatory=$true)][string]$Output)
Add-Type -AssemblyName System.Drawing
$out=[IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
$bmp=[Drawing.Bitmap]::new(1800,880)
$g=[Drawing.Graphics]::FromImage($bmp)
try {
  $g.SmoothingMode=[Drawing.Drawing2D.SmoothingMode]::AntiAlias
  $g.TextRenderingHint=[Drawing.Text.TextRenderingHint]::ClearTypeGridFit
  $g.Clear([Drawing.Color]::White)
  $title=[Drawing.Font]::new('Microsoft YaHei',28,[Drawing.FontStyle]::Bold)
  $body=[Drawing.Font]::new('Microsoft YaHei',17,[Drawing.FontStyle]::Regular)
  $small=[Drawing.Font]::new('Microsoft YaHei',14,[Drawing.FontStyle]::Regular)
  $bold=[Drawing.Font]::new('Microsoft YaHei',17,[Drawing.FontStyle]::Bold)
  $dark=[Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(35,55,75))
  $enc=[Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(214,234,248))
  $dec=[Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(220,242,225))
  $bott=[Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(244,224,229))
  $io=[Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(245,238,214))
  $border=[Drawing.Pen]::new([Drawing.Color]::FromArgb(70,95,115),3)
  $arrow=[Drawing.Pen]::new([Drawing.Color]::FromArgb(70,95,115),4)
  $arrow.CustomEndCap=[Drawing.Drawing2D.AdjustableArrowCap]::new(7,9,$true)
  $skip=[Drawing.Pen]::new([Drawing.Color]::FromArgb(234,132,62),3)
  $skip.DashStyle=[Drawing.Drawing2D.DashStyle]::Dash
  $skip.CustomEndCap=[Drawing.Drawing2D.AdjustableArrowCap]::new(6,8,$true)
  $center=[Drawing.StringFormat]::new(); $center.Alignment='Center'; $center.LineAlignment='Center'
  $g.DrawString('M4-P：ResNet34-U-Net 二维病灶分割结构',$title,$dark,[Drawing.RectangleF]::new(0,20,1800,55),$center)
  function Box($x,$y,$w,$h,$fill,$line1,$line2){
    $r=[Drawing.RectangleF]::new($x,$y,$w,$h); $g.FillRectangle($fill,$r); $g.DrawRectangle($border,$x,$y,$w,$h)
    $g.DrawString($line1,$bold,$dark,[Drawing.RectangleF]::new($x+4,$y+8,$w-8,36),$center)
    $g.DrawString($line2,$small,$dark,[Drawing.RectangleF]::new($x+4,$y+42,$w-8,$h-48),$center)
  }
  $xs=@(35,255,475,695,915,1135,1355,1575); $y=340; $w=185; $h=125
  Box $xs[0] $y $w $h $io '三通道输入' '256×256×3'
  Box $xs[1] $y $w $h $enc 'Stem' '7×7 Conv, 64'
  Box $xs[2] $y $w $h $enc 'Layer 1' 'ResNet34, 64'
  Box $xs[3] $y $w $h $enc 'Layer 2' 'ResNet34, 128'
  Box $xs[4] $y $w $h $bott 'Layer 3–4' '256 → 512'
  Box $xs[5] $y $w $h $dec '解码器 1–2' '上采样 + 拼接'
  Box $xs[6] $y $w $h $dec '解码器 3–5' '恢复至 256×256'
  Box $xs[7] $y $w $h $io '输出层' '1×1 Conv + Sigmoid'
  for($i=0;$i -lt 7;$i++){ $g.DrawLine($arrow,$xs[$i]+$w,$y+$h/2,$xs[$i+1]-8,$y+$h/2) }
  $g.DrawString('编码器：ImageNet 预训练 ResNet34',$body,$dark,[Drawing.RectangleF]::new(245,505,850,40),$center)
  $g.DrawString('解码器：转置卷积上采样 + DoubleConv',$body,$dark,[Drawing.RectangleF]::new(1110,505,620,40),$center)
  # Skip connections from encoder feature maps to decoder stages.
  $pairs=@(@(2,6,180),@(3,6,235),@(4,5,290),@(1,6,125))
  foreach($p in $pairs){
    $x1=$xs[$p[0]]+$w/2; $x2=$xs[$p[1]]+$w/2; $top=$p[2]
    $g.DrawLine($skip,$x1,$y,$x1,$top); $g.DrawLine($skip,$x1,$top,$x2,$top); $g.DrawLine($skip,$x2,$top,$x2,$y-5)
  }
  $g.DrawString('橙色虚线：对应尺度的跳跃连接（特征拼接）',$body,[Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(200,90,30)),[Drawing.RectangleF]::new(560,95,680,40),$center)
  $g.DrawString('前向路径：Stem → MaxPool → Layer1–4 → 五级上采样解码 → 单通道病灶概率图',$body,$dark,[Drawing.RectangleF]::new(260,620,1280,45),$center)
  $g.DrawString('训练配置：BCEWithLogits + Positive Dice；阈值 0.5；M4-P 使用轻量增强与 ImageNet 初始化',$small,$dark,[Drawing.RectangleF]::new(180,690,1440,45),$center)
  $g.DrawString('结构依据：src/brain_tumor_seg/model.py 中 ResNet34UNet 实现',$small,[Drawing.Brushes]::DimGray,[Drawing.RectangleF]::new(350,770,1100,35),$center)
  $dir=[IO.Path]::GetDirectoryName($out); if(-not(Test-Path $dir)){New-Item -ItemType Directory -Force $dir|Out-Null}
  $bmp.Save($out,[Drawing.Imaging.ImageFormat]::Png)
} finally { $g.Dispose(); $bmp.Dispose() }
$out
