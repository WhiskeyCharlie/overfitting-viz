# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

prefix = 'overfitting_venv/Lib/site-packages/'

a = Analysis(['src/app.py'],
             pathex=['../overfitting_venv/Lib/site-packages', '../Overfitting'],
             binaries=[],
             datas=[
             ('assets', 'assets'),
             ('assets/base-styles.css', 'base-styles.css'),
             ('assets/custom-styles.css', 'custom-styles.css'),
             (prefix + 'dash_core_components/', 'dash_core_components'),
             (prefix + 'dash_html_components/', 'dash_html_components'),
             (prefix + 'dash_bootstrap_components/', 'dash_bootstrap_components'),
             (prefix + 'dash', 'dash/')
             ],
             hiddenimports=['sklearn.utils._weight_vector'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='vizibly',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None,
          icon='assets/favicon.ico')
