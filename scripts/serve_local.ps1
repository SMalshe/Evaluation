# Serve the local models directly with llama.cpp's OpenAI-compatible server.
#
# One process per model, each pinned to an explicit context size and thread
# count. That is the point: Ollama chooses those itself and has repeatedly
# loaded a model at 64k context, claiming far more RAM than this machine has
# and pushing it into disk swap. Here the footprint is fixed and predictable,
# and because every model gets its own resident process there is no weight
# reloading between grid cells.
#
# Weights are the GGUF blobs Ollama already downloaded - no second copy.
#
#   pwsh scripts/serve_local.ps1            # start
#   pwsh scripts/serve_local.ps1 -Stop      # stop

param([switch]$Stop)

$BIN   = Join-Path $PSScriptRoot "..\.tools\cpu\llama-server.exe"
$BLOBS = "C:\Users\Lab User\SAIL\Ollama\blobs"

# port -> (blob digest, label). Threads are 6: measured best on the 12-core
# Oryon CPU, and it leaves headroom for a second model generating concurrently.
$MODELS = @(
    @{ Port = 8081; Label = "qwen2.5-14b"; Blob = "sha256-2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54" },
    @{ Port = 8082; Label = "llama3.1-8b"; Blob = "sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29" },
    @{ Port = 8083; Label = "llama3.2-3b"; Blob = "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff" },
    @{ Port = 8084; Label = "llama3.2-1b"; Blob = "sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45" }
)

if ($Stop) {
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    "stopped all llama-server processes"
    return
}

foreach ($m in $MODELS) {
    $path = Join-Path $BLOBS $m.Blob
    if (-not (Test-Path $path)) { Write-Error "missing weights for $($m.Label): $path"; continue }
    # The weights path contains a space ("Lab User"), and Start-Process joins
    # ArgumentList entries with spaces without quoting - so quote it here or the
    # path arrives split in half.
    $args = @(
        "-m", "`"$path`"",
        "--host", "127.0.0.1", "--port", $m.Port,
        "-c", "4096",          # hard context cap - the whole reason for this script
        "-t", "6",             # generation threads
        "-tb", "12",           # batch/prompt threads: prompt processing scales wider
        "--no-webui",
        "--alias", $m.Label
    )
    Start-Process -FilePath $BIN -ArgumentList $args -WindowStyle Hidden
    "started $($m.Label) on port $($m.Port)"
}

Start-Sleep -Seconds 20
foreach ($m in $MODELS) {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:$($m.Port)/health" -TimeoutSec 10 | Out-Null
        "  ok  $($m.Label)  :$($m.Port)"
    } catch {
        "  WAIT $($m.Label)  :$($m.Port)  (still loading weights)"
    }
}
