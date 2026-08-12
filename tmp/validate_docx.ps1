param(
    [Parameter(Mandatory = $true)]
    [string]$Path
)

Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression
$resolved = (Resolve-Path -LiteralPath $Path).Path
$archive = [System.IO.Compression.ZipFile]::OpenRead($resolved)
try {
    $names = @($archive.Entries | ForEach-Object FullName)
    $required = @('[Content_Types].xml', '_rels/.rels', 'word/document.xml', 'word/styles.xml')
    foreach ($name in $required) {
        if ($names -notcontains $name) { throw "Missing required part: $name" }
    }

    $xmlCount = 0
    $totalBytes = 0L
    foreach ($entry in $archive.Entries) {
        $stream = $entry.Open()
        $memory = [System.IO.MemoryStream]::new()
        try {
            $stream.CopyTo($memory)
            $bytes = $memory.ToArray()
            $totalBytes += $bytes.Length
            if ($entry.FullName.EndsWith('.xml') -or $entry.FullName.EndsWith('.rels')) {
                $text = [System.Text.Encoding]::UTF8.GetString($bytes)
                $doc = [System.Xml.XmlDocument]::new()
                $doc.PreserveWhitespace = $true
                $doc.LoadXml($text)
                $xmlCount++
            }
        }
        finally {
            $memory.Dispose()
            $stream.Dispose()
        }
    }

    [pscustomobject]@{
        File = $resolved
        Entries = $archive.Entries.Count
        XmlPartsParsed = $xmlCount
        MediaParts = @($archive.Entries | Where-Object FullName -like 'word/media/*').Count
        UncompressedBytesRead = $totalBytes
        PackageStatus = 'OK'
    }
}
finally {
    $archive.Dispose()
}
