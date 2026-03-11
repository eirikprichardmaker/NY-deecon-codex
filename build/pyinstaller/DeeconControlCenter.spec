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
    manifest='''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0"
          xmlns:asmv3="urn:schemas-microsoft-com:asm.v3">
  <assemblyIdentity type="win32" name="DeeconControlCenter"
                    version="1.0.0.0" processorArchitecture="*"/>
  <asmv3:application>
    <asmv3:windowsSettings>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">
        PerMonitorV2, PerMonitor
      </dpiAwareness>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">
        True/PM
      </dpiAware>
    </asmv3:windowsSettings>
  </asmv3:application>
</assembly>''',
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
