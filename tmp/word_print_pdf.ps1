param([Parameter(Mandatory=$true)][string]$Path,[Parameter(Mandatory=$true)][string]$Pdf)
$ErrorActionPreference='Stop'
$docx=(Resolve-Path -LiteralPath $Path).Path
$pdf=[IO.Path]::GetFullPath((Join-Path (Get-Location) $Pdf))
$word=$null;$doc=$null
try{
  $word=New-Object -ComObject Word.Application;$word.Visible=$false;$word.DisplayAlerts=0
  $doc=$word.Documents.Open($docx,$false,$true)
  $doc.PrintOut($false,$false,0,$pdf,'Microsoft Print to PDF',$false,$null,$null,'0',$null,$null,1,$true,$true,$null,$false)
  $limit=(Get-Date).AddSeconds(90)
  while((Get-Date)-lt$limit -and -not(Test-Path -LiteralPath $pdf)){Start-Sleep -Milliseconds 500}
  if(-not(Test-Path -LiteralPath $pdf)){throw 'PDF was not created by print driver'}
  [pscustomobject]@{Pdf=$pdf;Bytes=(Get-Item $pdf).Length}
}finally{if($doc){$doc.Close(0)};if($word){$word.Quit()};[GC]::Collect();[GC]::WaitForPendingFinalizers()}
