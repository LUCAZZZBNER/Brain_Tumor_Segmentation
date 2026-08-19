param(
  [Parameter(Mandatory=$true)][string]$Path,
  [Parameter(Mandatory=$true)][string]$OutputDir
)
$ErrorActionPreference='Stop'
$docx=(Resolve-Path -LiteralPath $Path).Path
$out=[IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDir))
if(-not(Test-Path -LiteralPath $out)){New-Item -ItemType Directory -Path $out | Out-Null}
$word=$null;$doc=$null;$scratch=$null
try{
  $word=New-Object -ComObject Word.Application;$word.Visible=$false;$word.DisplayAlerts=0
  $doc=$word.Documents.Open($docx,$false,$true)
  $doc.Repaginate()
  $pages=$doc.ComputeStatistics(2)
  for($i=1;$i -le $pages;$i++){
    $start=$doc.GoTo(1,1,$i).Start
    if($i -lt $pages){$end=$doc.GoTo(1,1,$i+1).Start-1}else{$end=$doc.Content.End-1}
    $range=$doc.Range($start,$end)
    $range.CopyAsPicture()
    $scratch=$word.Documents.Add()
    $scratch.Range(0,0).PasteSpecial($false,$false,0,$false,9)
    $shape=$scratch.InlineShapes.Item(1)
    $shape.SaveAsPicture((Join-Path $out ('page-{0:D2}.png' -f $i)))
    $scratch.Close(0);$scratch=$null
  }
  [pscustomobject]@{Docx=$docx;Pages=$pages;Images=(Get-ChildItem $out -Filter 'page-*.png').Count}
}finally{if($scratch){$scratch.Close(0)};if($doc){$doc.Close(0)};if($word){$word.Quit()};[GC]::Collect();[GC]::WaitForPendingFinalizers()}
