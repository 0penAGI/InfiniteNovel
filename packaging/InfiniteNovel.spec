# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build specification for Infinite Novel.
# macOS  -> InfiniteNovel.app (bundled via BUNDLE)
# Windows -> InfiniteNovel.exe + windowed build (used by installer)
import os
import sys

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))
ICON_ICNS = os.path.join(ROOT, "packaging", "icon.icns")
ICON_ICO = os.path.join(ROOT, "packaging", "icon.ico")

datas = []
binaries = []
hiddenimports = []


def _collect(name):
    global datas, binaries, hiddenimports
    try:
        d, b, h = collect_all(name)
    except Exception:
        return
    datas += d
    binaries += b
    hiddenimports += h


# Heavy AI / media stacks. collect_all pulls submodules + runtime data files
# (TTS model configs, transformers tokenizer data, diffusers schedulers, ...).
for pkg in [
    "TTS", "transformers", "diffusers", "torch", "torchvision",
    "torchaudio", "librosa", "cv2", "moviepy", "scipy", "numpy",
    "PIL", "requests", "pygame", "imageio", "imageio_ffmpeg",
    "safetensors", "tokenizers", "accelerate", "huggingface_hub",
    "soundfile", "yaml", "tqdm", "regex", "emoji", "numba", "llvmlite",
    "sklearn", "scipy", "audioread", "packaging",
    "gruut", "gruut_ipa", "gruut_lang_de", "gruut_lang_en",
    "gruut_lang_es", "gruut_lang_fr", "jamo", "unidic_lite", "num2words",
]:
    _collect(pkg)

# Game bundles its own source module (loaded at runtime by launcher) + media.
datas += [
    (os.path.join(ROOT, "infinite_novel.py"), "."),
    (os.path.join(ROOT, "intro.mp4"), "."),
]

a = Analysis(
    [os.path.join(ROOT, "packaging", "launcher.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[os.path.join(ROOT, "packaging", "rthook_typeguard.py")],
    excludes=["pytest", "IPython", "jupyter", "IPython.core.interactiveshell"],
    noarchive=False,
)

pyz = PYZ(a.pure)

if sys.platform == "darwin":
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="InfiniteNovel",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon=ICON_ICNS if os.path.exists(ICON_ICNS) else None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name="InfiniteNovel",
    )
    app = BUNDLE(
        coll,
        name="InfiniteNovel.app",
        icon=ICON_ICNS if os.path.exists(ICON_ICNS) else None,
        bundle_identifier="org.0penagi.infinite-novel",
        info_plist={
            "CFBundleName": "Infinite Novel",
            "CFBundleDisplayName": "Infinite Novel",
            "CFBundleShortVersionString": "0.2.0",
            "CFBundleVersion": "0.2.0",
            "CFBundlePackageType": "APPL",
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "© 0penAGI – Infinite Novel",
            "NSMicrophoneUsageDescription": "Infinite Novel uses the microphone for optional voice input.",
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="InfiniteNovel",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon=ICON_ICO if os.path.exists(ICON_ICO) else None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name="InfiniteNovel",
    )