# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SpyMeet Recorder — single-file .exe"""

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['recorder_app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'pyaudiowpatch',
        'pystray',
        'pystray._win32',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'soundfile',
        'numpy',
        'scipy',
        'scipy.signal',
        'audio_player',
        'pipeline_runner',
        'diagnostics_window',
        'queue',
        'threading',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'whisperx', 'torch', 'torchaudio', 'openai', 'groq', 'anthropic',
        'llm_process', 'noisereduce', 'pyloudnorm',
        'matplotlib', 'pandas', 'jupyter', 'IPython', 'notebook',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SpyMeetRecorder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,           # show console for debug output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,              # TODO: add .ico file
)
