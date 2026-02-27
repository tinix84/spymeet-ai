# Whisper Transcriber — Windows + GPU

Pipeline completa: trascrizione audio → correzione LLM → summary riunione.
WhisperX (Whisper large-v3) + Claude (Anthropic API).

---

## Struttura del progetto

```
D:\OneDrive\claude\spymeet\
├── check_gpu.ps1       ← STEP 1: rileva GPU e CUDA
├── setup.ps1           ← STEP 2: installa tutto
├── run.ps1             ← STEP 3: lancia la pipeline
├── transcribe.py       ← motore trascrizione (WhisperX)
├── llm_process.py      ← motore LLM (correzione + summary)
├── glossary.txt        ← opzionale: termini tecnici/nomi
└── audio\              ← metti qui i file audio/video
    └── transcripts\    ← output (creato automaticamente)
```

---

## Setup (una tantum)

### Prerequisiti
- Python 3.10+: https://www.python.org/downloads/
  - Spunta "Add Python to PATH" durante l'installazione
- GPU NVIDIA: driver aggiornati da https://www.nvidia.com/drivers

### Step 1 — Apri PowerShell come utente normale
```
Win + X → Windows PowerShell
```
Naviga nella cartella del progetto:
```powershell
cd D:\OneDrive\claude\spymeet
```

### Step 2 — Abilita script PowerShell (una volta)
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Step 3 — Rileva GPU
```powershell
.\check_gpu.ps1
```
Stampa la tua GPU, versione CUDA, e salva `gpu_config.json`.

### Step 4 — Installa tutto
```powershell
.\setup.ps1
```
Installa automaticamente: ffmpeg, PyTorch (versione GPU corretta), whisperx, anthropic.

### Step 5 — Configura le chiavi

**Anthropic API key** (Claude Code la imposta già):
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

**HuggingFace token** (per diarizzazione speaker):
1. Registrati: https://huggingface.co
2. Accetta: https://huggingface.co/pyannote/speaker-diarization-3.1
3. Accetta: https://huggingface.co/pyannote/segmentation-3.0
4. Token: https://huggingface.co/settings/tokens
```powershell
$env:HF_TOKEN = "hf_..."
```

Per rendere le chiavi permanenti, aggiungile al profilo PowerShell:
```powershell
notepad $PROFILE
# Aggiungi:
# $env:ANTHROPIC_API_KEY = "sk-ant-..."
# $env:HF_TOKEN = "hf_..."
```

---

## Utilizzo

### Pipeline completa (trascrivi + correggi + summary)
```powershell
.\run.ps1 -Language it
.\run.ps1 -Language de
.\run.ps1 -Language en
.\run.ps1               # auto-detect lingua
```

### Solo trascrizione (senza LLM)
```powershell
.\run.ps1 -Language it -SkipLLM
```

### Solo LLM su trascrizioni esistenti
```powershell
.\run.ps1 -LLMOnly -Input .\audio\transcripts
```

### Cartella input personalizzata
```powershell
.\run.ps1 -Language it -Input "D:\registrazioni\riunione_marzo"
```

### Con glossario tecnico
```powershell
.\run.ps1 -Language it -Glossary .\glossary.txt
```

### Modello più veloce (meno preciso)
```powershell
.\run.ps1 -Language it -Model medium
```

---

## Output

Per ogni file audio, in `audio\transcripts\`:

| File | Contenuto |
|------|-----------|
| `riunione.txt` | Trascrizione grezza con speaker e timestamp |
| `riunione.json` | JSON completo con timestamp per parola |
| `riunione_corrected.txt` | Trascrizione pulita, ⚠ sui segmenti dubbi |
| `riunione_summary.md` | Summary strutturata in Markdown |

---

## Glossario (opzionale)

Crea `glossary.txt` con termini tecnici, un termine per riga:
```
inverter
SRF-PLL
THD = Total Harmonic Distortion
V2G = Vehicle to Grid
TRU = Transport Refrigeration Unit
```

---

## Troubleshooting

**"running scripts is disabled"**
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**CUDA not available dopo installazione**
- Reinstalla driver NVIDIA dal sito ufficiale
- Riavvia e riesegui `check_gpu.ps1`

**whisperx non si installa**
```powershell
pip install whisperx --no-build-isolation
```

**Errore pyannote / diarizzazione**
- Verifica di aver accettato i termini su HuggingFace
- Funziona anche senza: la trascrizione gira uguale, senza label speaker

**Memoria GPU esaurita (CUDA out of memory)**
```powershell
.\run.ps1 -Language it -Model medium   # usa meno VRAM
```
