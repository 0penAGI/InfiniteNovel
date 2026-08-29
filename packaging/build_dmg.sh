#!/usr/bin/env bash
# Packages the built InfiniteNovel.app into a professional .dmg installer.
set -euo pipefail

VERSION="${1:-0.2.0}"
DIST="dist"
APP_NAME="Infinite Novel"

if [[ ! -d "$DIST/InfiniteNovel.app" ]]; then
    echo "ERROR: $DIST/InfiniteNovel.app not found. Run pyinstaller first."
    exit 1
fi

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
cp -R "$DIST/InfiniteNovel.app" "$STAGING/$APP_NAME.app"
ln -s /Applications "$STAGING/Applications"

DMG="$DIST/InfiniteNovel-${VERSION}-macOS-arm64.dmg"
rm -f "$DMG"
hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$STAGING" \
    -ov -format UDZO \
    "$DMG"

echo "Created: $DMG"