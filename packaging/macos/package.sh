#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ! "$3" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Usage: $0 <publish-dir> <output-dir> <MAJOR.MINOR.PATCH>" >&2
  exit 2
fi

publish_dir="$(cd "$1" && pwd)"
test -s "$publish_dir/ONNXStudioUI"
test -s "$publish_dir/libonnxruntime.dylib"
mkdir -p "$2"
output_dir="$(cd "$2" && pwd)"
version="$3"
staging_dir="$(mktemp -d)"
trap 'rm -rf "$staging_dir"' EXIT
app_dir="$staging_dir/ONNX Studio.app"
contents_dir="$app_dir/Contents"

mkdir -p "$contents_dir/MacOS" "$contents_dir/Resources"
cp -R "$publish_dir"/. "$contents_dir/MacOS/"
chmod 0755 "$contents_dir/MacOS/ONNXStudioUI"
cp "$publish_dir/Assets/onnxstudio.icns" "$contents_dir/Resources/onnxstudio.icns"

cat > "$contents_dir/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "https://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleDisplayName</key><string>ONNX Studio</string>
  <key>CFBundleExecutable</key><string>ONNXStudioUI</string>
  <key>CFBundleIdentifier</key><string>net.gailloty.onnxstudio</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>ONNX Studio</string>
  <key>CFBundleIconFile</key><string>onnxstudio.icns</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

plutil -lint "$contents_dir/Info.plist"
# Ad-hoc signing checks bundle integrity; Developer ID signing and notarization
# require separate Apple credentials and are not configured by this workflow.
codesign --force --deep --sign - "$app_dir"
codesign --verify --deep --strict "$app_dir"
ditto -c -k --sequesterRsrc --keepParent "$app_dir" "$output_dir/ONNXStudio-$version-osx-arm64.zip"
mkdir -p "$staging_dir/dmg"
cp -R "$app_dir" "$staging_dir/dmg/"
ln -s /Applications "$staging_dir/dmg/Applications"
hdiutil create -quiet -volname "ONNX Studio" -srcfolder "$staging_dir/dmg" \
  -ov -format UDZO "$output_dir/ONNXStudio-$version-osx-arm64.dmg"
