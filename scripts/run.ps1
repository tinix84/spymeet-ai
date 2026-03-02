# run.ps1
# Launch transcription + LLM pipeline.
# Must run scripts/setup.ps1 first.
#
# Examples (run from project root):
#   .\scripts\run.ps1 -Backend cpu -Language it
#   .\scripts\run.ps1 -Backend openai-api -Language it
#   .\scripts\run.ps1 -Backend openai-api -Language de -Input D:\recordings
#   .\scripts\run.ps1 -Backend cpu -Language it -SkipLLM
#   .\scripts\run.ps1 -LLMOnly -Input .\audio\transcripts

param(
    [string]$Backend  = "cpu",         # cpu | openai-api
    [string]$Input    = "",
    [string]$Output   = "",
    [string]$Language = "",            # it, de, en - empty = auto-detect
    [string]$Model    = "large-v3",    # WhisperX model (cpu backend only)
    [string]$HfToken  = "",            # HuggingFace token (cpu + diarization)
    [string]$Glossary = "",            # domain glossary .txt
    [switch]$SkipLLM,                  # skip LLM correction + summary
    [switch]$LLMOnly                   # run LLM only on existing .txt files
)

$ErrorActionPreference = "Stop"

# Resolve project root (one level up from scripts/)
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
if (-not $Input) { $Input = "$ProjectRoot\audio" }

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Whisper Transcriber - Run                " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Activate conda environment
$condaEnv = "social_env"
try {
    conda activate $condaEnv 2>$null
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
} catch {
    Write-Host "Failed to activate conda env '$condaEnv'." -ForegroundColor Red
    exit 1
}

# Check keys
if (-not $SkipLLM -and -not $LLMOnly -and -not $env:ANTHROPIC_API_KEY) {
    Write-Host "WARNING: ANTHROPIC_API_KEY not set - LLM step will be skipped." -ForegroundColor Yellow
    Write-Host '  $env:ANTHROPIC_API_KEY = "sk-ant-..."' -ForegroundColor Yellow
    Write-Host ""
}

if ($Backend -eq "openai-api" -and -not $env:OPENAI_API_KEY) {
    Write-Host "ERROR: OPENAI_API_KEY not set." -ForegroundColor Red
    Write-Host '  $env:OPENAI_API_KEY = "sk-..."' -ForegroundColor White
    exit 1
}

if ($Backend -eq "groq-api" -and -not $env:GROQ_API_KEY) {
    Write-Host "ERROR: GROQ_API_KEY not set." -ForegroundColor Red
    Write-Host "  Get free key at: https://console.groq.com" -ForegroundColor Yellow
    Write-Host '  $env:GROQ_API_KEY = "gsk_..."' -ForegroundColor White
    exit 1
}

if ($Backend -eq "cpu") {
    if ($HfToken) { $env:HF_TOKEN = $HfToken }
    if (-not $env:HF_TOKEN) {
        Write-Host "INFO: HF_TOKEN not set - speaker diarization skipped." -ForegroundColor Yellow
        Write-Host '  $env:HF_TOKEN = "hf_..."' -ForegroundColor Yellow
        Write-Host ""
    }
}

# Build and run command
if ($LLMOnly) {
    Write-Host "Mode: LLM only" -ForegroundColor Green
    $cmdArgs = @("$ProjectRoot\llm_process.py", "--input", $Input)
    if ($Glossary) { $cmdArgs += @("--glossary", $Glossary) }
    if ($Output)   { $cmdArgs += @("--output",   $Output)   }
    Write-Host "  python $($cmdArgs -join ' ')" -ForegroundColor DarkGray
    Write-Host ""
    & python @cmdArgs
} else {
    Write-Host "Mode: $Backend transcription + LLM" -ForegroundColor Green
    $cmdArgs = @("$ProjectRoot\transcribe.py", "--input", $Input, "--backend", $Backend)
    if ($Language) { $cmdArgs += @("--language", $Language) }
    if ($Output)   { $cmdArgs += @("--output",   $Output)   }
    if ($Glossary) { $cmdArgs += @("--glossary", $Glossary) }
    if ($SkipLLM)  { $cmdArgs += "--skip-llm" }
    if ($Backend -eq "cpu") {
        $cmdArgs += @("--model", $Model)
        if ($env:HF_TOKEN) { $cmdArgs += @("--hf-token", $env:HF_TOKEN) }
    }
    Write-Host "  python $($cmdArgs -join ' ')" -ForegroundColor DarkGray
    Write-Host ""
    & python @cmdArgs
}

Write-Host ""
Write-Host "Done. Outputs in: $Input\transcripts" -ForegroundColor Cyan
Write-Host ""
