#!/usr/bin/env python3
# Infinite Novel desktop launcher (PyInstaller entry point).
# Prepares the runtime environment, makes sure the local Ollama Gemma model
# is available, then boots the game from the bundled copy of infinite_novel.py.
import asyncio
import importlib.util
import os
import shutil
import subprocess
import sys
import threading


def bundle_dir():
    """Directory where bundled game assets (intro.mp4, infinite_novel.py) live."""
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def find_ollama():
    candidates = [
        shutil.which("ollama"),
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
        os.path.expanduser("~/.local/bin/ollama"),
        "/Applications/Ollama.app/Contents/Resources/ollama",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def ensure_ollama_model():
    """Best-effort background setup: pull the Gemma model used for dialogue."""
    try:
        ollama = find_ollama()
        if not ollama:
            return
        # Make sure the Ollama daemon is running before pulling.
        subprocess.run(
            [ollama, "list"], check=False, capture_output=True, timeout=30
        )
        subprocess.run(
            [ollama, "pull", "gemma3:1b"],
            check=False,
            capture_output=False,
            timeout=1800,
        )
    except Exception:
        pass


def main():
    # PyTorch MPS is not a full CUDA replacement; let unsupported ops fall back.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    root = bundle_dir()
    os.chdir(root)

    threading.Thread(target=ensure_ollama_model, daemon=True).start()

    game_path = os.path.join(root, "infinite_novel.py")
    spec = importlib.util.spec_from_file_location("infinite_novel", game_path)
    game = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(game)

    asyncio.run(game.main())


if __name__ == "__main__":
    main()