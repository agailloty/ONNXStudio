#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ! "$3" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Usage: $0 <publish-dir> <output-dir> <MAJOR.MINOR.PATCH>" >&2
  exit 2
fi

publish_dir="$(cd "$1" && pwd)"
test -s "$publish_dir/ONNXStudioUI"
test -s "$publish_dir/libonnxruntime.so"
mkdir -p "$2"
output_dir="$(cd "$2" && pwd)"
version="$3"
staging_dir="$(mktemp -d)"
trap 'rm -rf "$staging_dir"' EXIT

mkdir -p "$staging_dir/DEBIAN" "$staging_dir/opt/onnxstudio" \
  "$staging_dir/usr/bin" "$staging_dir/usr/share/applications"
cp -R "$publish_dir"/. "$staging_dir/opt/onnxstudio/"
chmod 0755 "$staging_dir/opt/onnxstudio/ONNXStudioUI"
ln -s /opt/onnxstudio/ONNXStudioUI "$staging_dir/usr/bin/onnxstudio"

installed_size="$(du -sk "$staging_dir/opt/onnxstudio" | cut -f1)"
cat > "$staging_dir/DEBIAN/control" <<EOF
Package: onnxstudio
Version: $version
Section: science
Priority: optional
Architecture: amd64
Installed-Size: $installed_size
Maintainer: agailloty <agailloty@users.noreply.github.com>
Homepage: https://github.com/agailloty/ONNXStudio
Depends: libc6 (>= 2.35), libgcc-s1, libstdc++6, zlib1g, libssl3 | libssl3t64, libicu70 | libicu72 | libicu74 | libicu76 | libicu78, libgssapi-krb5-2, libfontconfig1, libx11-6, libice6, libsm6
Description: Inspect, run and serve ONNX models
 ONNX Studio provides a computation graph explorer, model inference and
 an embedded REST API. The .NET runtime is included.
EOF

cat > "$staging_dir/usr/share/applications/onnxstudio.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=ONNX Studio
Comment=Inspect, run and serve ONNX models
Exec=/opt/onnxstudio/ONNXStudioUI
Terminal=false
Categories=Development;Science;
EOF

chmod 0755 "$staging_dir/DEBIAN"
chmod 0644 "$staging_dir/DEBIAN/control" "$staging_dir/usr/share/applications/onnxstudio.desktop"
dpkg-deb --root-owner-group --build "$staging_dir" "$output_dir/ONNXStudio-$version-linux-x64.deb"
tar -C "$staging_dir/opt/onnxstudio" -czf "$output_dir/ONNXStudio-$version-linux-x64.tar.gz" .
