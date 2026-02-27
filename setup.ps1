# setup.ps1
# Full automated setup for Whisper Transcriber on Windows.
# Reads gpu_config.json produced by check_gpu.ps1.
# Creates venv, installs PyTorch (GPU or CPU), whisperx, anthropic.

param(
    [string]$VenvDir = "venv",
    [switch]$Force   # re-install even if venv exists
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Whisper Transcriber — Windows Setup      " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── 0. Check execution policy ─────────────────────────────────────────────────
$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq "Restricted") {
    Write-Host "PowerShell execution policy is Restricted." -ForegroundColor Red
    Write-Host "Run this first (once):" -ForegroundColor Yellow
    Write-Host "  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned" -ForegroundColor White
    exit 1
}

# ── 1. Read GPU config ────────────────────────────────────────────────────────
$device = "cpu"
$torchIndex = $null

if (Test-Path "gpu_config.json") {
    $config = Get-Content "gpu_config.json" | ConvertFrom-Json
    $device = $config.device
    $torchIndex = $config.torch_index
    Write-Host "[0] GPU config loaded: device=$device" -ForegroundColor Green
} else {
    Write-Host "[0] gpu_config.json not found — run check_gpu.ps1 first." -ForegroundColor Yellow
    Write-Host "    Defaulting to CPU mode." -ForegroundColor Yellow
}

# ── 2. Check Python ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[1] Checking Python..." -ForegroundColor Yellow
$pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "    Python not found. Download: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
$pyVersion = & python --version 2>&1
Write-Host "    $pyVersion" -ForegroundColor Green

# ── 3. Check ffmpeg ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2] Checking ffmpeg..." -ForegroundColor Yellow
$ffmpegCmd = Get-Command "ffmpeg" -ErrorAction SilentlyContinue
if (-not $ffmpegCmd) {
    Write-Host "    ffmpeg not found. Installing via winget..." -ForegroundColor Yellow
    try {
        winget install ffmpeg --accept-package-agreements --accept-source-agreements
        Write-Host "    ffmpeg installed. You may need to restart PowerShell." -ForegroundColor Green
    } catch {
        Write-Host "    winget failed. Install manually: https://ffmpeg.org/download.html" -ForegroundColor Red
        Write-Host "    Then add ffmpeg/bin to your PATH and restart PowerShell." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "    ffmpeg OK: $($ffmpegCmd.Source)" -ForegroundColor Green
}

# ── 4. Create virtual environment ─────────────────────────────────────────────
Write-Host ""
Write-Host "[3] Creating virtual environment in .\$VenvDir ..." -ForegroundColor Yellow

if (Test-Path $VenvDir) {
    if ($Force) {
        Write-Host "    Removing existing venv (--Force)..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VenvDir
    } else {
        Write-Host "    venv already exists. Use -Force to recreate." -ForegroundColor Green
    }
}

if (-not (Test-Path $VenvDir)) {
    & python -m venv $VenvDir
    Write-Host "    Created." -ForegroundColor Green
}

# ── 5. Activate venv ──────────────────────────────────────────────────────────
$activateScript = ".\$VenvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "[4] Activating venv..." -ForegroundColor Yellow
& $activateScript
Write-Host "    Active." -ForegroundColor Green

# ── 6. Upgrade pip ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[5] Upgrading pip..." -ForegroundColor Yellow
& python -m pip install --upgrade pip --quiet

# ── 7. Install PyTorch ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[6] Installing PyTorch..." -ForegroundColor Yellow

if ($device -eq "cuda" -and $torchIndex) {
    Write-Host "    GPU mode: $torchIndex" -ForegroundColor Green
    & pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$torchIndex" --quiet
} else {
    Write-Host "    CPU mode" -ForegroundColor Yellow
    & pip install torch torchaudio --quiet
}
Write-Host "    PyTorch installed." -ForegroundColor Green

# ── 8. Install whisperx ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "[7] Installing whisperx..." -ForegroundColor Yellow
& pip install whisperx --quiet
Write-Host "    whisperx installed." -ForegroundColor Green

# ── 9. Install anthropic SDK ──────────────────────────────────────────────────
Write-Host ""
Write-Host "[8] Installing anthropic + openai SDKs..." -ForegroundColor Yellow
& pip install "anthropic>=0.30.0" "openai>=1.0.0" "groq>=0.9.0" --quiet
Write-Host "    anthropic + openai + groq installed." -ForegroundColor Green

# ── 9b. Install audio enhancement dependencies ──────────────────────────────
Write-Host ""
Write-Host "[8b] Installing audio enhancement dependencies..." -ForegroundColor Yellow
& pip install "pyloudnorm>=0.1.0" "noisereduce>=3.0.0" "scipy>=1.10.0" "soundfile>=0.12.0" --quiet
Write-Host "     pyloudnorm + noisereduce + scipy + soundfile installed." -ForegroundColor Green

# ── 10. Verify GPU is visible to PyTorch ──────────────────────────────────────
Write-Host ""
Write-Host "[9] Verifying PyTorch GPU access..." -ForegroundColor Yellow
$torchCheck = & python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" 2>&1
Write-Host "    $torchCheck" -ForegroundColor Green

# ── 11. Check ANTHROPIC_API_KEY ───────────────────────────────────────────────
Write-Host ""
Write-Host "[10] Checking ANTHROPIC_API_KEY..." -ForegroundColor Yellow
$apiKey = $env:ANTHROPIC_API_KEY
if ($apiKey) {
    Write-Host "    ANTHROPIC_API_KEY found (${apiKey.Substring(0, [Math]::Min(12, $apiKey.Length))}...)" -ForegroundColor Green
} else {
    Write-Host "    ANTHROPIC_API_KEY not set." -ForegroundColor Red
    Write-Host "    Set it now for this session:" -ForegroundColor Yellow
    Write-Host '    $env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"' -ForegroundColor White
    Write-Host "    To persist across sessions, add to your PowerShell profile:" -ForegroundColor Yellow
    Write-Host '    Add-Content $PROFILE ''$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"''' -ForegroundColor White
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup complete!                          " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Put audio files in .\audio\" -ForegroundColor White
Write-Host "  2. Run: .\run.ps1 -Language it" -ForegroundColor White
Write-Host "  3. Find outputs in .\audio\transcripts\" -ForegroundColor White
Write-Host ""
Write-Host "For HuggingFace diarization token setup, see README.md" -ForegroundColor Yellow
Write-Host ""
