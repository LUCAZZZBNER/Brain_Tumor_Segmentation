param([Parameter(Mandatory=$true)][string]$Path)
Add-Type -AssemblyName System.IO.Compression.FileSystem
$a=[IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $Path))
try {
  $e=$a.GetEntry('word/document.xml'); $sr=[IO.StreamReader]::new($e.Open())
  try {[xml]$x=$sr.ReadToEnd()} finally {$sr.Dispose()}
  $n=[Xml.XmlNamespaceManager]::new($x.NameTable)
  $n.AddNamespace('w','http://schemas.openxmlformats.org/wordprocessingml/2006/main')
  $n.AddNamespace('a','http://schemas.openxmlformats.org/drawingml/2006/main')
  $n.AddNamespace('v','urn:schemas-microsoft-com:vml')
  $body=$x.SelectSingleNode('//w:body',$n)
  $lastText=''; $tableNo=0; $imageNo=0; $index=0
  foreach($node in $body.ChildNodes){
    if($node.LocalName -eq 'p'){
      $text=(($node.SelectNodes('.//w:t',$n)|% InnerText)-join '')
      $drawings=$node.SelectNodes('.//w:drawing | .//w:pict',$n).Count
      if($drawings -gt 0){$imageNo += $drawings; "IMAGE#$imageNo body=$index previous=[$lastText] current=[$text]"}
      if($text){$lastText=$text}
    } elseif($node.LocalName -eq 'tbl'){
      $tableNo++; $rows=$node.SelectNodes('./w:tr',$n).Count; $cols=if($rows){$node.SelectNodes('./w:tr[1]/w:tc',$n).Count}else{0}
      "TABLE#$tableNo body=$index rows=$rows cols=$cols previous=[$lastText]"
    }
    $index++
  }
  "TOTAL tables=$tableNo images=$imageNo"
} finally {$a.Dispose()}
