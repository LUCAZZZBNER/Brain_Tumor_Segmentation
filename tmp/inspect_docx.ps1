param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [string]$Pattern = ''
)

Add-Type -AssemblyName System.IO.Compression.FileSystem
$resolved = (Resolve-Path -LiteralPath $Path).Path
$archive = [System.IO.Compression.ZipFile]::OpenRead($resolved)
try {
    $entry = $archive.GetEntry('word/document.xml')
    if (-not $entry) { throw 'word/document.xml not found' }
    $reader = [System.IO.StreamReader]::new($entry.Open())
    try { [xml]$xml = $reader.ReadToEnd() } finally { $reader.Dispose() }

    $ns = [System.Xml.XmlNamespaceManager]::new($xml.NameTable)
    $ns.AddNamespace('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
    $paragraphs = $xml.SelectNodes('//w:body//w:p', $ns)
    for ($i = 0; $i -lt $paragraphs.Count; $i++) {
        $p = $paragraphs[$i]
        $styleNode = $p.SelectSingleNode('./w:pPr/w:pStyle', $ns)
        $style = if ($styleNode) { $styleNode.GetAttribute('val', $ns.LookupNamespace('w')) } else { '' }
        $textNodes = $p.SelectNodes('.//w:t | .//w:tab | .//w:br', $ns)
        $parts = foreach ($node in $textNodes) {
            if ($node.LocalName -eq 't') { $node.InnerText }
            elseif ($node.LocalName -eq 'tab') { "`t" }
            else { '<BR>' }
        }
        $text = ($parts -join '')
        if (-not $Pattern -or $text -match $Pattern -or $style -match $Pattern) {
            '{0:D4}`t{1}`t{2}' -f $i, $style, $text
        }
    }
}
finally {
    $archive.Dispose()
}
