param(
  [Parameter(Mandatory=$true)][string]$Path,
  [Parameter(Mandatory=$true)][string]$OutputDir
)
$ErrorActionPreference='Stop'
$docx=(Resolve-Path -LiteralPath $Path).Path
$out=[IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDir))
if(-not(Test-Path -LiteralPath $out)){New-Item -ItemType Directory -Path $out | Out-Null}
$pdf=Join-Path $out (([IO.Path]::GetFileNameWithoutExtension($docx))+'.pdf')
$word=$null;$doc=$null
try{
  $word=New-Object -ComObject Word.Application;$word.Visible=$false;$word.DisplayAlerts=0
  $doc=$word.Documents.Open($docx,$false,$true)
  $doc.SaveAs2($pdf,17)
  $pages=$doc.ComputeStatistics(2)
  [pscustomobject]@{Docx=$docx;Pdf=$pdf;Pages=$pages;Bytes=(Get-Item $pdf).Length}
}finally{if($doc){$doc.Close(0)};if($word){$word.Quit()};[GC]::Collect();[GC]::WaitForPendingFinalizers()}
