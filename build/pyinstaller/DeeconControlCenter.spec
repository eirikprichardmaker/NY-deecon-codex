# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

datas = [('G:\\Min disk\\NEW DEECON\\config', 'config'), ('G:\\Min disk\\NEW DEECON\\configs', 'configs')]
hiddenimports = ['pytest']
datas += collect_data_files('bs4')
hiddenimports += collect_submodules('src')
hiddenimports += collect_submodules('bs4')


a = Analysis(
    ['..\\..\\src\\gui_windows_entry.py'],
    pathex=['G:\\Min disk\\NEW DEECON'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DeeconControlCenter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DeeconControlCenter',
)
