# Build and release

The [release workflow](../.github/workflows/release.yml) follows
[TidyMemo's build and packaging pipeline](https://github.com/agailloty/TidyMemo/blob/master/.github/workflows/dotnet.yml).
It runs on every pushed tag and can also be started manually from GitHub Actions.

## Publish a version

Commit and push the workflow and packaging files before tagging the desired commit:

```sh
git tag v1.0.0
git push origin v1.0.0
```

Tags must be `vMAJOR.MINOR.PATCH` or `MAJOR.MINOR.PATCH`, without leading zeroes
or prerelease suffixes. Each number must be at most 65535 for Windows version
metadata. Invalid tags fail validation before any build starts. A local tag only
starts the workflow when it is pushed to GitHub.

The workflow restores and builds the solution, runs the Core/API and headless UI
tests on every target OS, then publishes the real `ONNXStudioUI` application in
Release mode. It includes .NET, ASP.NET Core and the native ONNX Runtime library;
users do not need to install .NET. Trimming and single-file publishing are disabled
to preserve runtime dependencies and reflection used by the application.

| Target | Release files |
| --- | --- |
| Windows x64 | `ONNXStudio-<version>-win-x64.zip`, `ONNXStudio-<version>-win-x64-Setup.exe` |
| Linux x64 | `ONNXStudio-<version>-linux-x64.tar.gz`, `ONNXStudio-<version>-linux-x64.deb` |
| macOS Apple Silicon | `ONNXStudio-<version>-osx-arm64.zip`, `ONNXStudio-<version>-osx-arm64.dmg` |

Windows installs per user with Start menu shortcuts and an uninstaller. On macOS,
drag `ONNX Studio.app` to Applications. The Debian package installs under
`/opt/onnxstudio` and adds a desktop entry and the `onnxstudio` command. Portable
Windows/Linux archives contain `ONNXStudioUI.exe` / `ONNXStudioUI` at their root.
Linux still requires the system libraries listed in the DEB's dependencies.

Only after all three builds and their tests succeed does the workflow create a
GitHub Release, attach the six packages and `checksums-sha256.txt`, and generate
release notes. Test reports and packages are also kept as Actions artifacts for
14 days. Only the release job receives `contents: write`, using the built-in
`GITHUB_TOKEN`; no additional repository secret is needed.

**Run workflow** performs the same validation and packaging without creating a
release, even when manually run against a tag. Branch builds use version
`0.0.<run-number>`. This is useful for checking all three platforms before tagging.
To retry a failed build, rerun the failed jobs. If a release already exists for a
tag, release creation fails rather than replacing its published assets; use a new
version for changed binaries.

## Supported targets and signing

The current `Microsoft.ML.OnnxRuntime` 1.30.0 package includes `osx-arm64`, but no
`osx-x64` native runtime. Intel macOS is therefore excluded from the build matrix.
The macOS bundle declares macOS 14 or newer.

Packages are not publisher-signed. The macOS bundle has an ad-hoc signature for
integrity, not a Developer ID signature or notarization. Production signing can
be added later using protected GitHub secrets.

## Local validation

Use a stable .NET 10 SDK (`global.json` selects the latest installed stable 10.0
feature band):

```sh
dotnet restore ONNXStudio.slnx
dotnet build ONNXStudio.slnx -c Release --no-restore -p:UsedAvaloniaProducts=
dotnet test tests/ONNXStudio.Core.Tests -c Release --no-build --no-restore
dotnet test tests/ONNXStudio.UI.Tests -c Release --no-build --no-restore
dotnet publish ONNXStudioUI/ONNXStudioUI.csproj -c Release -r win-x64 --self-contained true -o publish/win-x64 -p:Version=1.0.0 -p:PublishTrimmed=false -p:PublishSingleFile=false -p:UsedAvaloniaProducts=
```

`UsedAvaloniaProducts=` disables Avalonia's build telemetry. Change the runtime
and output directory for another supported target. Native packaging runs on its
target OS: Windows requires NSIS, Linux uses `dpkg-deb` and `tar`, and macOS uses
`codesign`, `ditto`, `plutil` and `hdiutil`. Bash scripts accept the publish directory,
output directory and numeric version, for example:

```sh
bash packaging/linux/package.sh publish/linux-x64 artifacts 1.0.0
bash packaging/macos/package.sh publish/osx-arm64 artifacts 1.0.0
```

The logo master is `ONNXStudioUI/Assets/onnxstudio.svg`. To regenerate the PNG,
multi-resolution Windows ICO and macOS ICNS on Windows, run
`powershell -NoProfile -ExecutionPolicy Bypass -File packaging/Generate-BrandAssets.ps1`.
These assets are copied into the publish output and used by all three installers.
