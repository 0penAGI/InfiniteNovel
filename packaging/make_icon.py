#!/usr/bin/env python3
# Builds a branded app icon (icon.icns for macOS, icon.ico for Windows)
# from the project banner assets.
import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = os.path.dirname(os.path.abspath(__file__))
BANNER = os.path.join(os.path.dirname(ROOT), "novel.png")
if not os.path.exists(BANNER):
    BANNER = os.path.join(os.path.dirname(ROOT), "nolev.png")
SIZE = 1024


def build_square_icon() -> Image.Image:
    banner = Image.open(BANNER).convert("RGB")
    banner = ImageOps.fit(banner, (SIZE, int(SIZE * 0.24)), method=Image.LANCZOS)

    canvas = Image.new("RGB", (SIZE, SIZE), (11, 11, 22))

    # subtle vertical gradient
    overlay = Image.new("RGB", (SIZE, SIZE))
    d = ImageDraw.Draw(overlay)
    for y in range(SIZE):
        t = y / SIZE
        r = int(11 + 30 * t)
        g = int(11 + 26 * t)
        b = int(22 + 40 * t)
        d.line([(0, y), (SIZE, y)], fill=(r, g, b))
    canvas = Image.blend(canvas, overlay, 0.35)

    # soft glow behind the banner
    glow = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse(
        [SIZE // 2 - 420, SIZE // 2 - 220, SIZE // 2 + 420, SIZE // 2 + 220],
        fill=(60, 40, 120),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    canvas = Image.blend(canvas, glow, 0.35)

    # center banner, add thin border
    canvas.paste(banner, (0, (SIZE - banner.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, SIZE - 1, SIZE - 1], outline=(120, 120, 190), width=3)

    return canvas


def write_icns(icon: Image.Image, out: str):
    icon.save(os.path.join(ROOT, "icon_1024.png"))
    iconset = os.path.join(ROOT, "icon.iconset")
    if os.path.exists(iconset):
        shutil.rmtree(iconset)
    os.makedirs(iconset, exist_ok=True)
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for size in sizes:
        resized = icon.resize((size, size), Image.LANCZOS)
        for name in (f"icon_{size}x{size}.png",):
            pass
        resized.save(os.path.join(iconset, f"icon_{size}x{size}.png"))
        if size in (32, 64, 256, 512, 1024):
            resized.save(os.path.join(iconset, f"icon_{size // 2}x{size // 2}@2x.png"))
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out], check=True)
    shutil.rmtree(iconset)
    os.unlink(os.path.join(ROOT, "icon_1024.png"))


def write_ico(icon: Image.Image, out: str):
    icon.save(out, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])


def main():
    icon = build_square_icon()
    if sys.platform == "darwin":
        write_icns(icon, os.path.join(ROOT, "icon.icns"))
    write_ico(icon, os.path.join(ROOT, "icon.ico"))
    print("icons generated:", os.path.join(ROOT, "icon.icns"), os.path.join(ROOT, "icon.ico"))


if __name__ == "__main__":
    main()