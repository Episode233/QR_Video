# -*- mode: python ; coding: utf-8 -*-
dll_path="E:/qr/.venv/Lib/site-packages/pyzbar/"

a = Analysis(
    ['decode.py'],
    pathex=[],
    binaries=[(dll_path+'libiconv.dll','.'),(dll_path+'libzbar-64.dll','.'),],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('v', None, 'OPTION')],
    name='decode',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
