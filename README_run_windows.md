# DeeconControlCenter Windows Build

This project can be packaged as a Windows GUI app so end users do not need a local Python install.

## Build

Run from the repository root (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

Optional clean build:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 -Clean
```

The script will:
1. Create a virtual environment for build tools.
2. Install `requirements.txt`.
3. Install PyInstaller.
4. Build a one-dir app with no console window.

## Output

Expected output executable:

`dist\DeeconControlCenter\DeeconControlCenter.exe`

The build also includes:
- `config\` (so default `config\config.yaml` works)
- `configs\`
- `README_run_windows.md`
- bundled Python runtime and imported `src` modules

At runtime, the app creates/uses `runs\` inside `dist\DeeconControlCenter\`.

## Run (end user)

1. Open `dist\DeeconControlCenter\`.
2. Start `DeeconControlCenter.exe`.

The app sets its working directory to the EXE folder in frozen mode, so relative paths like `config\config.yaml` and `runs\` resolve inside `dist\DeeconControlCenter\`.
