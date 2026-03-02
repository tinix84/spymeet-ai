# check_gpu.ps1
# Detects GPU type, CUDA version, and prints the correct PyTorch install command.
# Run first before scripts/setup.ps1
#
# Run from project root: .\scripts\check_gpu.ps1

# Resolve project root (one level up from scripts/)
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  GPU / CUDA Detection for Whisper Setup   " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Detect GPU via WMI
Write-Host "[1] Detecting GPU..." -ForegroundColor Yellow
$gpus = Get-WmiObject Win32_VideoController | Select-Object -ExpandProperty Name
foreach ($gpu in $gpus) {
    Write-Host "    Found: $gpu"
}

$isNvidia = $gpus | Where-Object { $_ -match "NVIDIA" }
$isAmd    = $gpus | Where-Object { $_ -match "AMD|Radeon" }
$isIntel  = $gpus | Where-Object { $_ -match "Intel" }

Write-Host ""

# 2. Check nvidia-smi
$cudaVersion = $null
if ($isNvidia) {
    Write-Host "[2] NVIDIA GPU found - checking CUDA via nvidia-smi..." -ForegroundColor Yellow
    $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $smiOutput = & nvidia-smi 2>&1
        $cudaLine = $smiOutput | Select-String "CUDA Version"
        if ($cudaLine) {
            Write-Host "    $cudaLine" -ForegroundColor Green
            $cudaString = $cudaLine.ToString()
            if ($cudaString -match "CUDA Version: (\d+)") {
                $cudaMajor = $Matches[1]
                $cudaVersion = $cudaMajor
            }
        }
    } else {
        Write-Host "    nvidia-smi not found - CUDA drivers may not be installed." -ForegroundColor Red
        Write-Host "    Download CUDA from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
    }
} elseif ($isAmd) {
    Write-Host "[2] AMD GPU detected - ROCm not supported on Windows for WhisperX." -ForegroundColor Red
    Write-Host "    Will fall back to CPU mode." -ForegroundColor Yellow
} elseif ($isIntel) {
    Write-Host "[2] Intel GPU detected - will use CPU mode." -ForegroundColor Yellow
} else {
    Write-Host "[2] No discrete GPU detected - will use CPU mode." -ForegroundColor Yellow
}

Write-Host ""

# 3. Python check
Write-Host "[3] Python version..." -ForegroundColor Yellow
$pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
if ($pythonCmd) {
    $pyVersion = & python --version 2>&1
    Write-Host "    $pyVersion" -ForegroundColor Green
    if ($pyVersion -match "3\.(\d+)") {
        $pyMinor = [int]$Matches[1]
        if ($pyMinor -lt 10) {
            Write-Host "    WARNING: Python 3.10+ required. Download: https://www.python.org/downloads/" -ForegroundColor Red
        }
    }
} else {
    Write-Host "    Python not found. Download: https://www.python.org/downloads/" -ForegroundColor Red
}

Write-Host ""

# 4. ffmpeg check
Write-Host "[4] ffmpeg..." -ForegroundColor Yellow
$ffmpegCmd = Get-Command "ffmpeg" -ErrorAction SilentlyContinue
if ($ffmpegCmd) {
    Write-Host "    ffmpeg found: $($ffmpegCmd.Source)" -ForegroundColor Green
} else {
    Write-Host "    ffmpeg NOT found." -ForegroundColor Red
    Write-Host "    Install with: winget install ffmpeg" -ForegroundColor Yellow
    Write-Host "    Then restart PowerShell." -ForegroundColor Yellow
}

Write-Host ""

# 5. Recommended PyTorch install command
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  RECOMMENDED PYTORCH INSTALL COMMAND      " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

if ($isNvidia -and $cudaVersion) {
    $torchIndex = "cu121"
    $cudaLabel  = "12.x"
    if ($cudaVersion -eq "11") {
        $torchIndex = "cu118"
        $cudaLabel  = "11.x"
    }
    $torchCmd = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/$torchIndex"
    Write-Host "  CUDA $cudaLabel detected - use:" -ForegroundColor Green
    Write-Host ""
    Write-Host "  $torchCmd" -ForegroundColor White
    Write-Host ""
    Write-Host "  setup.ps1 will use this automatically." -ForegroundColor Green

    $config = [ordered]@{
        cuda_version = $cudaVersion
        torch_index  = $torchIndex
        device       = "cuda"
    }
    $config | ConvertTo-Json | Set-Content "$ProjectRoot\gpu_config.json"
    Write-Host "  Saved config to gpu_config.json" -ForegroundColor Green

} elseif ($isNvidia -and -not $cudaVersion) {
    Write-Host "  NVIDIA GPU found but CUDA not detected." -ForegroundColor Red
    Write-Host "  1. Install CUDA: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    Write-Host "  2. Reboot" -ForegroundColor Yellow
    Write-Host "  3. Re-run this script" -ForegroundColor Yellow

    $config = [ordered]@{ device = "cpu" }
    $config | ConvertTo-Json | Set-Content "$ProjectRoot\gpu_config.json"
} else {
    Write-Host "  No NVIDIA GPU - CPU mode" -ForegroundColor Yellow
    Write-Host "  pip install torch torchaudio" -ForegroundColor White
    Write-Host ""
    Write-Host "  NOTE: CPU mode is slow for long recordings." -ForegroundColor Yellow
    Write-Host "  A 1h recording may take 1-3h on CPU." -ForegroundColor Yellow

    $config = [ordered]@{ device = "cpu" }
    $config | ConvertTo-Json | Set-Content "$ProjectRoot\gpu_config.json"
}

Write-Host ""
Write-Host "Next step: run .\scripts\setup.ps1" -ForegroundColor Cyan
Write-Host ""
