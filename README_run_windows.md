# DeeconControlCenter (Windows, no Python needed)

Denne mappen beskriver hvordan du bygger GUI som en `.exe` med PyInstaller.

## Resultat av build

Etter kjøring av `scripts/build_windows.ps1` får du:

- `dist/DeeconControlCenter/DeeconControlCenter.exe`
- `dist/DeeconControlCenter/config/config.yaml`
- `dist/DeeconControlCenter/runs/`
- `dist/DeeconControlCenter/README_run_windows.md`

## Build på Windows (PowerShell)

```powershell
Set-Location <repo-root>
.\scripts\build_windows.ps1
```

Scriptet gjør automatisk:

1. Oppretter virtualenv (`.venv-windows-build`)
2. Installerer avhengigheter (`pip install -r requirements.txt`)
3. Installerer PyInstaller
4. Bygger GUI i one-folder mode uten konsollvindu

## Kjøring

Fra `dist/DeeconControlCenter`:

- Dobbeltklikk `DeeconControlCenter.exe`
- Eller fra PowerShell:

```powershell
Set-Location .\dist\DeeconControlCenter
.\DeeconControlCenter.exe
```

GUI og intern pipeline kjører med arbeidsmappe satt til exe-mappen, så relative paths fungerer:

- `config\config.yaml`
- `runs\<run_id>\...`

## Feilsøking

- Hvis build feiler på manglende pakker: kjør scriptet på nytt, det oppgraderer `pip` og installerer avhengigheter på nytt.
- Hvis `config\config.yaml` mangler i dist-mappen: bygg på nytt og verifiser at `config/` finnes i repo-roten.
