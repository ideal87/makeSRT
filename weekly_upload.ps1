<#
Weekly caption upload (headless): fetch -> shift -> upload.

  1. fetch_sermon_m4a.py --srt   downloads the service .m4a and the *_final_en_TTS.srt
  2. shift_srt.py --cuts auto    finds the short YouTube video and fits the SRT to it
  3. upload_captions.py          uploads the shifted SRT as the video's English track

Stops at the first step that exits non-zero and exits with that step's code
(3 = sermon/m4a/SRT not on the site yet, so a scheduler can retry later).
Everything, including each step's progress output, goes to downloads\weekly_YYYYMMDD.log.

Usage:
    powershell -ExecutionPolicy Bypass -File weekly_upload.ps1                  # today's sermon
    powershell -ExecutionPolicy Bypass -File weekly_upload.ps1 -Date 20261004
    powershell -ExecutionPolicy Bypass -File weekly_upload.ps1 -DryRun          # don't touch YouTube
#>
param(
    [ValidatePattern('^\d{8}$')]
    [string]$Date = (Get-Date -Format 'yyyyMMdd'),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'   # Korean filenames in the scripts' output

New-Item -ItemType Directory -Force 'downloads' | Out-Null
$LogPath = Join-Path $PSScriptRoot "downloads\weekly_$Date.log"

function Write-Log([string]$Message) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

# Run one python step, copy its stdout/stderr into the log, and return its stdout lines.
# A non-zero exit code ends the whole script with that code.
function Invoke-Step([string]$Name, [string[]]$PyArgs) {
    Write-Log "--- $Name"
    $outFile = [IO.Path]::GetTempFileName()
    $errFile = [IO.Path]::GetTempFileName()
    try {
        $quoted = $PyArgs | ForEach-Object { '"' + $_ + '"' }
        $proc = Start-Process -FilePath 'python' -ArgumentList $quoted -NoNewWindow -Wait -PassThru `
                              -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $stdout = @(Get-Content -Path $outFile -Encoding UTF8)
        $stderr = @(Get-Content -Path $errFile -Encoding UTF8)
    } finally {
        Remove-Item $outFile, $errFile -ErrorAction SilentlyContinue
    }
    foreach ($l in $stderr + $stdout) { if ($l) { Add-Content -Path $LogPath -Value "    $l" -Encoding UTF8 } }

    if ($proc.ExitCode -ne 0) {
        $why = if ($proc.ExitCode -eq 3) { 'not ready yet, retry later' } else { 'failed' }
        Write-Log "$Name $why (exit $($proc.ExitCode)). Stopping."
        foreach ($l in $stderr | Select-Object -Last 5) { Write-Host "    $l" }
        exit $proc.ExitCode
    }
    return $stdout
}

Write-Log "=== Weekly upload for $Date$(if ($DryRun) { ' (dry run)' })"

# 1. Fetch: stdout is the m4a path, then the SRT path
$paths = @(Invoke-Step 'fetch' @('fetch_sermon_m4a.py', '--date', $Date, '--srt') | Where-Object { $_ })
if ($paths.Count -lt 2) { Write-Log "fetch printed $($paths.Count) path(s), expected 2. Stopping."; exit 2 }
$m4a, $srt = $paths[0], $paths[1]
Write-Log "m4a: $m4a"
Write-Log "srt: $srt"

# 2. Shift onto the short video; stdout has 'URL:' and 'Cuts:' lines
$shifted = [IO.Path]::ChangeExtension($srt, $null).TrimEnd('.') + '_shifted.srt'
$shiftOut = Invoke-Step 'shift' @('shift_srt.py', $m4a, $srt, '--cuts', 'auto', '--out', $shifted)
foreach ($l in $shiftOut) { if ($l -match '^(Video|URL|Cuts):') { Write-Log $l } }
$url = ($shiftOut | Where-Object { $_ -match '^URL:\s*(\S+)' } | Select-Object -First 1) -replace '^URL:\s*', ''
if (-not $url) { Write-Log 'shift_srt.py did not report a video URL. Stopping.'; exit 2 }

# 3. Upload
$uploadArgs = @('upload_captions.py', $shifted, $url)
if ($DryRun) { $uploadArgs += '--dry-run' }
Invoke-Step 'upload' $uploadArgs | Out-Null

Write-Log "=== Done: $url$(if ($DryRun) { ' (dry run, nothing uploaded)' })"
exit 0
