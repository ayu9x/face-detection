# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Face Detection Suite.
Bundles all models and Haar cascade files into a single .exe.
"""

import os
import cv2

# Find OpenCV's haarcascades directory
haar_dir = os.path.join(os.path.dirname(cv2.__file__), 'data')

a = Analysis(
    ['face.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Bundle all model files
        ('models/*.caffemodel', 'models'),
        ('models/*.prototxt', 'models'),
        # Bundle OpenCV haar cascades
        (os.path.join(haar_dir, 'haarcascade_frontalface_default.xml'), 'cv2/data'),
        (os.path.join(haar_dir, 'haarcascade_eye_tree_eyeglasses.xml'), 'cv2/data'),
        (os.path.join(haar_dir, 'haarcascade_smile.xml'), 'cv2/data'),
    ],
    hiddenimports=['numpy', 'cv2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FaceDetectionSuite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,
)
