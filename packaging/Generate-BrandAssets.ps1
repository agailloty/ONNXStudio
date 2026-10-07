# Rebuild the application icons from the vector master. Run on Windows with
# PowerShell: ./packaging/Generate-BrandAssets.ps1
# Only the SVG primitives used by this logo are needed; no external tools.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$assets = Join-Path $PSScriptRoot '../ONNXStudioUI/Assets'
[xml]$svg = Get-Content -LiteralPath (Join-Path $assets 'onnxstudio.svg') -Raw

function New-LogoBrush([string]$fill) {
    if ($fill -match '^url\(#(.+)\)$') {
        $gradient = $svg.svg.defs.linearGradient | Where-Object { $_.id -eq $Matches[1] }
        return [System.Drawing.Drawing2D.LinearGradientBrush]::new(
            [System.Drawing.Point]::new(0, 0), [System.Drawing.Point]::new(256, 256),
            [System.Drawing.ColorTranslator]::FromHtml($gradient.stop[0].'stop-color'),
            [System.Drawing.ColorTranslator]::FromHtml($gradient.stop[1].'stop-color'))
    }
    return [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml($fill))
}

$master = [System.Drawing.Bitmap]::new(1024, 1024)
$graphics = [System.Drawing.Graphics]::FromImage($master)
$graphics.SmoothingMode = 'AntiAlias'
$graphics.ScaleTransform(4, 4)
foreach ($element in $svg.svg.ChildNodes) {
    $path = [System.Drawing.Drawing2D.GraphicsPath]::new()
    switch ($element.LocalName) {
        'rect' {
            $x = [float]$element.x; $y = [float]$element.y
            $w = [float]$element.width; $h = [float]$element.height; $d = 2 * [float]$element.rx
            $path.AddArc($x, $y, $d, $d, 180, 90)
            $path.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
            $path.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90)
            $path.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
            $path.CloseFigure()
            $brush = New-LogoBrush $element.fill
            $graphics.FillPath($brush, $path)
            $brush.Dispose()
        }
        'polyline' {
            [System.Drawing.PointF[]]$points = @($element.points -split ' ' | ForEach-Object {
                $xy = $_ -split ','
                [System.Drawing.PointF]::new([float]$xy[0], [float]$xy[1])
            })
            $path.AddLines($points)
            if ($points[0] -eq $points[-1]) { $path.CloseFigure() }
            $brush = New-LogoBrush $element.stroke
            $pen = [System.Drawing.Pen]::new($brush, [float]$element.'stroke-width')
            $pen.LineJoin = 'Round'; $pen.StartCap = 'Round'; $pen.EndCap = 'Round'
            $graphics.DrawPath($pen, $path)
            $pen.Dispose(); $brush.Dispose()
        }
        'circle' {
            $r = [float]$element.r
            $brush = New-LogoBrush $element.fill
            $graphics.FillEllipse($brush, [float]$element.cx - $r, [float]$element.cy - $r, 2 * $r, 2 * $r)
            $brush.Dispose()
        }
    }
    $path.Dispose()
}
$graphics.Dispose()

function Get-PngBytes([int]$size) {
    $bitmap = [System.Drawing.Bitmap]::new($size, $size)
    $g = [System.Drawing.Graphics]::FromImage($bitmap)
    $g.InterpolationMode = 'HighQualityBicubic'
    $g.PixelOffsetMode = 'HighQuality'
    $g.DrawImage($master, 0, 0, $size, $size)
    $stream = [System.IO.MemoryStream]::new()
    $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
    $bytes = $stream.ToArray()
    $g.Dispose(); $bitmap.Dispose(); $stream.Dispose()
    return ,$bytes
}

[System.IO.File]::WriteAllBytes((Join-Path $assets 'onnxstudio.png'), (Get-PngBytes 512))
$sizes = @(16, 24, 32, 48, 64, 128, 256)
$frames = @($sizes | ForEach-Object { ,(Get-PngBytes $_) })
$stream = [System.IO.MemoryStream]::new()
$writer = [System.IO.BinaryWriter]::new($stream)
$writer.Write([uint16]0); $writer.Write([uint16]1); $writer.Write([uint16]$sizes.Count)
$offset = 6 + 16 * $sizes.Count
for ($i = 0; $i -lt $sizes.Count; $i++) {
    $dimension = if ($sizes[$i] -eq 256) { 0 } else { $sizes[$i] }
    $writer.Write([byte]$dimension); $writer.Write([byte]$dimension)
    $writer.Write([byte]0); $writer.Write([byte]0)
    $writer.Write([uint16]1); $writer.Write([uint16]32)
    $writer.Write([uint32]$frames[$i].Length); $writer.Write([uint32]$offset)
    $offset += $frames[$i].Length
}
foreach ($frame in $frames) { $writer.Write([byte[]]$frame) }
[System.IO.File]::WriteAllBytes((Join-Path $assets 'onnxstudio.ico'), $stream.ToArray())
$writer.Dispose(); $stream.Dispose()

# ICNS stores each PNG representation in a chunk with a big-endian length.
function Write-BigEndian($writer, [int]$value) {
    $bytes = [BitConverter]::GetBytes($value)
    [Array]::Reverse($bytes)
    $writer.Write($bytes)
}
$stream = [System.IO.MemoryStream]::new()
$writer = [System.IO.BinaryWriter]::new($stream)
$writer.Write([System.Text.Encoding]::ASCII.GetBytes('icns'))
Write-BigEndian $writer 0
$icnsSizes = [ordered]@{ icp4 = 16; icp5 = 32; icp6 = 64; ic07 = 128; ic08 = 256; ic09 = 512; ic10 = 1024 }
foreach ($entry in $icnsSizes.GetEnumerator()) {
    $png = Get-PngBytes $entry.Value
    $writer.Write([System.Text.Encoding]::ASCII.GetBytes($entry.Key))
    Write-BigEndian $writer ($png.Length + 8)
    $writer.Write($png)
}
$length = $stream.Length
$stream.Position = 4
Write-BigEndian $writer $length
[System.IO.File]::WriteAllBytes((Join-Path $assets 'onnxstudio.icns'), $stream.ToArray())
$writer.Dispose(); $stream.Dispose(); $master.Dispose()
Write-Output 'Generated onnxstudio.png, onnxstudio.ico and onnxstudio.icns from onnxstudio.svg.'
