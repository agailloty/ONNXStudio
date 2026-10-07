using System.Formats.Tar;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Logging.Abstractions;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Python;
using Xunit;

namespace ONNXStudio.Core.Tests;

internal sealed class FakeProcessRunner : IProcessRunner
{
    private readonly Func<ProcessSpec, ProcessResult> _handler;
    public List<ProcessSpec> Calls { get; } = new();

    public FakeProcessRunner(Func<ProcessSpec, ProcessResult> handler) => _handler = handler;

    public Task<ProcessResult> RunAsync(ProcessSpec spec, CancellationToken cancellationToken = default)
    {
        Calls.Add(spec);
        return Task.FromResult(_handler(spec));
    }
}

internal sealed class FakeDownloader : IPythonDownloader
{
    public Dictionary<string, string> Texts { get; } = new();
    public byte[]? File { get; set; }
    public List<string> Requested { get; } = new();

    public Task<string> GetStringAsync(string url, CancellationToken cancellationToken = default)
    {
        Requested.Add(url);
        return Texts.TryGetValue(url, out var text) ? Task.FromResult(text) : throw new HttpRequestException("404 " + url);
    }

    public async Task DownloadFileAsync(string url, string destination, IProgress<double>? progress, CancellationToken cancellationToken = default)
    {
        Requested.Add(url);
        if (File == null) throw new HttpRequestException("offline");
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        await System.IO.File.WriteAllBytesAsync(destination, File, cancellationToken);
        progress?.Report(1);
    }
}

internal sealed class TempDirectory : IDisposable
{
    public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "onnxstudio-test-" + Guid.NewGuid().ToString("N"));

    public TempDirectory() => Directory.CreateDirectory(Path);

    public void Dispose()
    {
        try { Directory.Delete(Path, recursive: true); } catch (IOException) { }
    }
}

public class PythonDistributionResolverTests
{
    private static string Sums(string tag, string triple) => string.Join("\n",
        $"{new string('a', 64)}  cpython-3.11.17+{tag}-{triple}-install_only_stripped.tar.gz",
        $"{new string('b', 64)}  cpython-3.12.4+{tag}-{triple}-install_only_stripped.tar.gz",
        $"{new string('c', 64)}  cpython-3.12.15+{tag}-{triple}-install_only_stripped.tar.gz",
        $"{new string('d', 64)}  cpython-3.12.15+{tag}-{triple}-install_only.tar.gz",
        $"{new string('e', 64)}  cpython-3.12.15+{tag}-{triple}-debug-full.tar.zst");

    [Fact]
    public void SelectsTheNewestMatchingStrippedBuildWithItsChecksum()
    {
        var result = PythonDistributionResolver.Select(Sums("20261003", "x86_64-pc-windows-msvc"), "20261003", "x86_64-pc-windows-msvc", "3.12");

        Assert.NotNull(result);
        Assert.Equal("3.12.15", result!.Version);
        Assert.Equal(new string('c', 64), result.Sha256);
        Assert.Equal(
            "https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.12.15%2B20261003-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
            result.Url.ToString());
    }

    [Fact]
    public void ReturnsNullWhenNoBuildMatches()
    {
        Assert.Null(PythonDistributionResolver.Select(Sums("1", "x"), "1", "other-triple", "3.12"));
        Assert.Null(PythonDistributionResolver.Select(Sums("1", "x"), "1", "x", "3.9"));
    }

    [Fact]
    public async Task UsesTheLatestReleaseTag()
    {
        var triple = PythonDistributionResolver.CurrentTriple();
        if (triple == null) return;
        var downloader = new FakeDownloader();
        downloader.Texts["https://raw.githubusercontent.com/astral-sh/python-build-standalone/latest-release/latest-release.json"] = "{\"tag\":\"20300101\"}";
        downloader.Texts["https://github.com/astral-sh/python-build-standalone/releases/download/20300101/SHA256SUMS"] = Sums("20300101", triple);

        var result = await new PythonDistributionResolver(downloader, new OnnxStudioPythonOptions()).ResolveAsync();

        Assert.True(result.IsSuccess, result.Error?.Message);
        Assert.Contains("/20300101/", result.Value!.Url.ToString());
    }

    [Fact]
    public async Task FallsBackToThePinnedReleaseWhenTheLatestTagCannotBeRead()
    {
        var triple = PythonDistributionResolver.CurrentTriple();
        if (triple == null) return;
        var options = new OnnxStudioPythonOptions { FallbackReleaseTag = "20250101" };
        var downloader = new FakeDownloader();
        downloader.Texts["https://github.com/astral-sh/python-build-standalone/releases/download/20250101/SHA256SUMS"] = Sums("20250101", triple);

        var result = await new PythonDistributionResolver(downloader, options).ResolveAsync();

        Assert.True(result.IsSuccess, result.Error?.Message);
        Assert.Contains("/20250101/", result.Value!.Url.ToString());
    }

    [Fact]
    public async Task ReportsAnActionableErrorWhenOffline()
    {
        if (PythonDistributionResolver.CurrentTriple() == null) return;
        var result = await new PythonDistributionResolver(new FakeDownloader(), new OnnxStudioPythonOptions()).ResolveAsync();

        Assert.True(result.IsFailure);
        Assert.Equal(PythonErrorCode.DownloadFailed, result.Error!.Code);
    }
}

public class PythonDiscoveryTests
{
    [Fact]
    public void ParsesPyLauncherOutput()
    {
        const string output = " -V:3.14[-64] *   C:\\Users\\me\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe\r\n -V:3.12          C:\\Python312\\python.exe\r\nnot a python line\r\n";

        var paths = PythonDiscovery.ParsePyLauncherOutput(output).ToArray();

        Assert.Equal(new[]
        {
            "C:\\Users\\me\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe",
            "C:\\Python312\\python.exe"
        }, paths);
    }

    [Fact]
    public void ResolvesInterpreterFilesAndEnvironmentFolders()
    {
        using var temp = new TempDirectory();
        var relative = OperatingSystem.IsWindows() ? Path.Combine("Scripts", "python.exe") : Path.Combine("bin", "python3");
        var executable = Path.Combine(temp.Path, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(executable)!);
        File.WriteAllText(executable, "");
        var discovery = new PythonDiscovery(new FakeProcessRunner(_ => new ProcessResult(1, "", "", false)));

        Assert.Equal(executable, discovery.ResolveExecutable(temp.Path));
        Assert.Equal(executable, discovery.ResolveExecutable("\"" + executable + "\""));
        Assert.Null(discovery.ResolveExecutable(Path.Combine(temp.Path, "missing")));
        Assert.Null(discovery.ResolveExecutable(""));
    }
}

public class PythonProbeTests
{
    private static ProcessResult Json(string version, string packages) =>
        new(0, $"noise from sitecustomize\n{{\"version\":\"{version}\",\"arch\":\"AMD64\",\"packages\":{packages}}}\n", "", false);

    [Fact]
    public async Task ReadsVersionAndPackagesIgnoringNoise()
    {
        using var temp = new TempDirectory();
        var exe = Path.Combine(temp.Path, "python");
        File.WriteAllText(exe, "");
        var probe = new PythonProbe(new FakeProcessRunner(_ => Json("3.12.4", "{\"numpy\":\"2.0.0\",\"scikit-learn\":\"1.5.0\",\"joblib\":\"1.4\",\"onnx\":null}")));

        var result = await probe.ProbeAsync(exe, PythonSource.System);

        Assert.NotNull(result);
        Assert.Equal("3.12.4", result!.Version);
        Assert.True(result.CanRunInference);
        Assert.False(result.CanConvert);
        Assert.Equal(new[] { "skl2onnx", "onnx" }, result.MissingForConversion);
    }

    [Fact]
    public async Task RejectsOldFailingAndMissingInterpreters()
    {
        using var temp = new TempDirectory();
        var exe = Path.Combine(temp.Path, "python");
        File.WriteAllText(exe, "");

        Assert.Null(await new PythonProbe(new FakeProcessRunner(_ => Json("3.8.10", "{}"))).ProbeAsync(exe, PythonSource.System));
        Assert.Null(await new PythonProbe(new FakeProcessRunner(_ => new ProcessResult(1, "", "boom", false))).ProbeAsync(exe, PythonSource.System));
        Assert.Null(await new PythonProbe(new FakeProcessRunner(_ => new ProcessResult(0, "not json", "", false))).ProbeAsync(exe, PythonSource.System));
        Assert.Null(await new PythonProbe(new FakeProcessRunner(_ => Json("3.12.0", "{}"))).ProbeAsync(Path.Combine(temp.Path, "nope"), PythonSource.System));
    }

    [Theory]
    [InlineData("3.9.0", true)]
    [InlineData("3.12.15", true)]
    [InlineData("3.8.18", false)]
    [InlineData("2.7.18", false)]
    [InlineData("3.13.0rc1", true)]
    public void ChecksTheMinimumVersion(string version, bool supported)
        => Assert.Equal(supported, PythonProbe.IsSupportedVersion(version));

    [Fact]
    public void ManagedRuntimeIgnoresUserSitePackages()
    {
        Assert.Equal("1", PythonEnvironment.For(PythonSource.Managed)["PYTHONNOUSERSITE"]);
        Assert.Null(PythonEnvironment.For(PythonSource.System)["PYTHONNOUSERSITE"]);
    }
}

public class PythonRuntimeInstallerTests
{
    private static (PythonPaths Paths, FakeDownloader Downloader, FakeProcessRunner Runner, PythonRuntimeInstaller Installer) Create(
        TempDirectory temp, byte[] archive, string? checksum = null)
    {
        var triple = PythonDistributionResolver.CurrentTriple() ?? "x86_64-pc-windows-msvc";
        var file = $"cpython-3.12.15+20260101-{triple}-install_only_stripped.tar.gz";
        var hash = checksum ?? Convert.ToHexString(SHA256.HashData(archive)).ToLowerInvariant();
        var downloader = new FakeDownloader { File = archive };
        downloader.Texts["https://raw.githubusercontent.com/astral-sh/python-build-standalone/latest-release/latest-release.json"] = "{\"tag\":\"20260101\"}";
        downloader.Texts["https://github.com/astral-sh/python-build-standalone/releases/download/20260101/SHA256SUMS"] = $"{hash}  {file}\n";

        var paths = new PythonPaths(Path.Combine(temp.Path, "py"));
        var runner = new FakeProcessRunner(_ => new ProcessResult(0, "", "", false));
        var installer = new PythonRuntimeInstaller(paths,
            new PythonDistributionResolver(downloader, new OnnxStudioPythonOptions()), downloader, runner,
            NullLogger<PythonRuntimeInstaller>.Instance);
        return (paths, downloader, runner, installer);
    }

    private static byte[] BuildArchive(PythonPaths paths)
    {
        var entry = Path.GetRelativePath(paths.ManagedRoot, paths.ManagedExecutable).Replace('\\', '/');
        using var buffer = new MemoryStream();
        using (var gzip = new GZipStream(buffer, CompressionLevel.Fastest, leaveOpen: true))
        using (var tar = new TarWriter(gzip))
        {
            var tarEntry = new PaxTarEntry(TarEntryType.RegularFile, entry) { DataStream = new MemoryStream(new byte[] { 1, 2, 3 }) };
            tar.WriteEntry(tarEntry);
        }
        return buffer.ToArray();
    }

    [Fact]
    public async Task DownloadsVerifiesAndExtractsThePythonArchive()
    {
        if (PythonDistributionResolver.CurrentTriple() == null) return;
        using var temp = new TempDirectory();
        var paths = new PythonPaths(Path.Combine(temp.Path, "py"));
        var (_, _, _, installer) = Create(temp, BuildArchive(paths));
        var progress = new List<PythonInstallProgress>();

        var result = await installer.InstallPythonAsync(new Progress<PythonInstallProgress>(progress.Add));

        Assert.True(result.IsSuccess, result.Error?.Message);
        Assert.Equal(paths.ManagedExecutable, result.Value);
        Assert.True(File.Exists(paths.ManagedExecutable));
        Assert.False(Directory.Exists(paths.ManagedRoot + ".staging"));
        Assert.Empty(Directory.GetFiles(paths.DownloadDirectory));
    }

    [Fact]
    public async Task RefusesAnArchiveWhoseChecksumDoesNotMatch()
    {
        if (PythonDistributionResolver.CurrentTriple() == null) return;
        using var temp = new TempDirectory();
        var paths = new PythonPaths(Path.Combine(temp.Path, "py"));
        var (_, _, _, installer) = Create(temp, BuildArchive(paths), checksum: new string('0', 64));

        var result = await installer.InstallPythonAsync(null);

        Assert.True(result.IsFailure);
        Assert.Equal(PythonErrorCode.ChecksumMismatch, result.Error!.Code);
        Assert.False(Directory.Exists(paths.ManagedRoot));
    }

    [Fact]
    public async Task ReportsDownloadFailures()
    {
        if (PythonDistributionResolver.CurrentTriple() == null) return;
        using var temp = new TempDirectory();
        var (_, downloader, _, installer) = Create(temp, Array.Empty<byte>());
        downloader.File = null;

        var result = await installer.InstallPythonAsync(null);

        Assert.Equal(PythonErrorCode.DownloadFailed, result.Error!.Code);
    }

    [Fact]
    public async Task InstallsPackagesWithPipUsingBinaryWheelsOnly()
    {
        using var temp = new TempDirectory();
        var (_, _, runner, installer) = Create(temp, Array.Empty<byte>());

        var result = await installer.InstallPackagesAsync("python", new[] { "numpy", "scikit-learn" }, null);

        Assert.True(result.IsSuccess);
        var call = Assert.Single(runner.Calls);
        Assert.Equal(new[] { "-m", "pip", "install", "--upgrade", "--only-binary", ":all:", "--no-input", "numpy", "scikit-learn" }, call.Arguments);
        Assert.Equal("1", call.Environment!["PYTHONNOUSERSITE"]);
    }

    [Fact]
    public async Task ReportsPipFailures()
    {
        using var temp = new TempDirectory();
        var (paths, downloader, _, _) = Create(temp, Array.Empty<byte>());
        var failing = new PythonRuntimeInstaller(paths, new PythonDistributionResolver(downloader, new OnnxStudioPythonOptions()), downloader,
            new FakeProcessRunner(_ => new ProcessResult(1, "", "ERROR: no matching distribution", false)), NullLogger<PythonRuntimeInstaller>.Instance);

        var result = await failing.InstallPackagesAsync("python", new[] { "numpy" }, null);

        Assert.Equal(PythonErrorCode.InstallFailed, result.Error!.Code);
        Assert.Contains("no matching distribution", result.Error.TechnicalDetails);
    }
}

public class PythonRuntimeServiceTests
{
    private sealed class FakeProbe : IPythonProbe
    {
        public Dictionary<string, PythonInterpreter?> Known { get; } = new(StringComparer.OrdinalIgnoreCase);

        public Task<PythonInterpreter?> ProbeAsync(string executablePath, PythonSource source, CancellationToken cancellationToken = default)
            => Task.FromResult(Known.TryGetValue(executablePath, out var i) && i != null
                ? new PythonInterpreter(i.ExecutablePath, i.Version, i.Architecture, source, i.Packages)
                : null);
    }

    private sealed class FakeDiscovery : IPythonDiscovery
    {
        public List<string> Candidates { get; } = new();
        public Task<IReadOnlyList<string>> FindCandidatesAsync(CancellationToken cancellationToken = default) => Task.FromResult<IReadOnlyList<string>>(Candidates);
        public string? ResolveExecutable(string pathOrFolder) => File.Exists(pathOrFolder) ? pathOrFolder : null;
    }

    private sealed class FakeInstaller : IPythonRuntimeInstaller
    {
        public Func<Result<string, PythonError>>? OnInstallPython { get; set; }
        public int PackageInstalls { get; private set; }
        public bool Removed { get; private set; }

        public Task<Result<string, PythonError>> InstallPythonAsync(IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default)
            => Task.FromResult(OnInstallPython!());

        public Task<Result<bool, PythonError>> InstallPackagesAsync(string executablePath, IReadOnlyList<string> packages,
            IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default)
        {
            PackageInstalls++;
            return Task.FromResult(Result<bool, PythonError>.Success(true));
        }

        public void RemoveManagedRuntime() => Removed = true;
    }

    private static PythonInterpreter Interpreter(string path, params string[] packages)
        => new(path, "3.12.4", "AMD64", PythonSource.System,
            PythonRequirements.Probed.ToDictionary(p => p, p => packages.Contains(p) ? "1.0" : null, StringComparer.OrdinalIgnoreCase));

    private static readonly string[] AllPackages = PythonRequirements.Managed.ToArray();

    private static (PythonRuntimeService Service, FakeProbe Probe, FakeDiscovery Discovery, FakeInstaller Installer, PythonPaths Paths) Create(TempDirectory temp)
    {
        var paths = new PythonPaths(Path.Combine(temp.Path, "py"));
        var probe = new FakeProbe();
        var discovery = new FakeDiscovery();
        var installer = new FakeInstaller();
        return (new PythonRuntimeService(paths, discovery, probe, installer, NullLogger<PythonRuntimeService>.Instance), probe, discovery, installer, paths);
    }

    private static string Touch(string directory, string name)
    {
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, name);
        File.WriteAllText(path, "");
        return path;
    }

    [Fact]
    public async Task AutomaticSelectionPrefersTheManagedRuntime()
    {
        using var temp = new TempDirectory();
        var (service, probe, discovery, _, paths) = Create(temp);
        var system = Touch(temp.Path, "sys-python");
        discovery.Candidates.Add(system);
        probe.Known[system] = Interpreter(system, AllPackages);
        Touch(Path.GetDirectoryName(paths.ManagedExecutable)!, Path.GetFileName(paths.ManagedExecutable));
        probe.Known[paths.ManagedExecutable] = Interpreter(paths.ManagedExecutable);

        var active = await service.GetActiveAsync();

        Assert.Equal(PythonSource.Managed, active!.Source);
    }

    [Fact]
    public async Task AutomaticSelectionFallsBackToASystemPythonThatHasEveryPackage()
    {
        using var temp = new TempDirectory();
        var (service, probe, discovery, _, _) = Create(temp);
        var incomplete = Touch(temp.Path, "a");
        var complete = Touch(temp.Path, "b");
        discovery.Candidates.AddRange(new[] { incomplete, complete });
        probe.Known[incomplete] = Interpreter(incomplete, "numpy");
        probe.Known[complete] = Interpreter(complete, AllPackages);

        var active = await service.GetActiveAsync();

        Assert.Equal(complete, active!.ExecutablePath);
        Assert.Equal(PythonSource.System, active.Source);
    }

    [Fact]
    public async Task NoRuntimeWhenNothingUsableExists()
    {
        using var temp = new TempDirectory();
        var (service, _, discovery, _, _) = Create(temp);
        discovery.Candidates.Add(Touch(temp.Path, "unprobeable"));

        Assert.Null(await service.GetActiveAsync());
    }

    [Fact]
    public async Task ExplicitSelectionIsPersistedAndRestored()
    {
        using var temp = new TempDirectory();
        var (service, probe, _, _, paths) = Create(temp);
        var custom = Touch(temp.Path, "custom-python");
        probe.Known[custom] = Interpreter(custom, AllPackages);
        var changes = 0;
        service.Changed += () => changes++;

        service.Select(new PythonSelection(custom, PythonSource.Custom));

        Assert.Equal(1, changes);
        Assert.Equal(custom, (await service.GetActiveAsync())!.ExecutablePath);
        Assert.Equal(PythonSource.Custom, (await service.GetActiveAsync())!.Source);

        var restored = new PythonRuntimeService(paths, new FakeDiscovery(), probe, new FakeInstaller(), NullLogger<PythonRuntimeService>.Instance);
        Assert.Equal(new PythonSelection(custom, PythonSource.Custom), restored.Selection);
    }

    [Fact]
    public void CorruptSelectionFileFallsBackToAutomatic()
    {
        using var temp = new TempDirectory();
        var paths = new PythonPaths(Path.Combine(temp.Path, "py"));
        Directory.CreateDirectory(paths.Root);
        File.WriteAllText(paths.SelectionFile, "{broken");

        var service = new PythonRuntimeService(paths, new FakeDiscovery(), new FakeProbe(), new FakeInstaller(), NullLogger<PythonRuntimeService>.Instance);

        Assert.True(service.Selection.IsAutomatic);
    }

    [Fact]
    public async Task ProbeCustomRejectsFoldersWithoutPython()
    {
        using var temp = new TempDirectory();
        var (service, _, _, _, _) = Create(temp);

        var result = await service.ProbeCustomAsync(Path.Combine(temp.Path, "nothing"));

        Assert.Equal(PythonErrorCode.InvalidInterpreter, result.Error!.Code);
    }

    [Fact]
    public async Task PackagesAreNeverInstalledWithoutTheManagedRuntime()
    {
        using var temp = new TempDirectory();
        var (service, _, _, installer, _) = Create(temp);

        var result = await service.InstallPackagesAsync(null);

        Assert.Equal(PythonErrorCode.NoRuntime, result.Error!.Code);
        Assert.Equal(0, installer.PackageInstalls);
    }

    [Fact]
    public async Task InstallingPythonThenPackagesMakesTheManagedRuntimeReady()
    {
        using var temp = new TempDirectory();
        var (service, probe, _, installer, paths) = Create(temp);
        installer.OnInstallPython = () =>
        {
            Touch(Path.GetDirectoryName(paths.ManagedExecutable)!, Path.GetFileName(paths.ManagedExecutable));
            probe.Known[paths.ManagedExecutable] = Interpreter(paths.ManagedExecutable);
            return Result<string, PythonError>.Success(paths.ManagedExecutable);
        };

        var python = await service.InstallPythonAsync(null);
        Assert.True(python.IsSuccess, python.Error?.Message);
        Assert.True(service.IsManagedInstalled);
        Assert.False(python.Value!.CanRunInference);

        probe.Known[paths.ManagedExecutable] = Interpreter(paths.ManagedExecutable, AllPackages);
        var packages = await service.InstallPackagesAsync(null);

        Assert.True(packages.IsSuccess);
        Assert.True(packages.Value!.CanConvert);
        Assert.Equal(1, installer.PackageInstalls);
    }

    [Fact]
    public async Task RemovingTheManagedRuntimeResetsAnExplicitManagedSelection()
    {
        using var temp = new TempDirectory();
        var (service, _, _, installer, paths) = Create(temp);
        service.Select(new PythonSelection(paths.ManagedExecutable, PythonSource.Managed));

        await service.RemoveManagedAsync();

        Assert.True(installer.Removed);
        Assert.True(service.Selection.IsAutomatic);
    }
}

public class PythonModelServiceTests
{
    private sealed class FakeRuntime : IPythonRuntimeService
    {
        public PythonInterpreter? Active { get; set; }
        public event Action? Changed { add { } remove { } }
        public PythonSelection Selection => PythonSelection.Automatic;
        public void Select(PythonSelection selection) { }
        public bool IsManagedInstalled => false;
        public Task<IReadOnlyList<PythonInterpreter>> DiscoverAsync(CancellationToken cancellationToken = default) => Task.FromResult<IReadOnlyList<PythonInterpreter>>(Array.Empty<PythonInterpreter>());
        public Task<Result<PythonInterpreter, PythonError>> ProbeCustomAsync(string pathOrFolder, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<PythonInterpreter?> GetActiveAsync(bool refresh = false, CancellationToken cancellationToken = default) => Task.FromResult(Active);
        public Task<Result<PythonInterpreter, PythonError>> InstallPythonAsync(IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<Result<PythonInterpreter, PythonError>> InstallPackagesAsync(IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task RemoveManagedAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    }

    private sealed class FakeWorker : IPythonWorkerClient
    {
        public JsonObject Response { get; set; } = new();
        public string? Command { get; private set; }
        public JsonObject? Request { get; private set; }

        public Task<Result<JsonObject, PythonError>> RunAsync(PythonInterpreter interpreter, string command, JsonObject request, CancellationToken cancellationToken = default)
        {
            Command = command;
            Request = request;
            return Task.FromResult(Result<JsonObject, PythonError>.Success(Response));
        }
    }

    private static PythonInterpreter Interpreter(PythonSource source, params string[] packages)
        => new("python", "3.12.4", "AMD64", source,
            PythonRequirements.Probed.ToDictionary(p => p, p => packages.Contains(p) ? "1.0" : null, StringComparer.OrdinalIgnoreCase));

    private static readonly string[] Everything = PythonRequirements.Managed.ToArray();

    private static string ModelFile(TempDirectory temp, string name = "model.joblib")
    {
        var path = Path.Combine(temp.Path, name);
        File.WriteAllText(path, "x");
        return path;
    }

    [Fact]
    public async Task EveryOperationRequiresTrustBeforeAnythingRuns()
    {
        using var temp = new TempDirectory();
        var worker = new FakeWorker();
        var service = new PythonModelService(new FakeRuntime { Active = Interpreter(PythonSource.Managed, Everything) }, worker);
        var path = ModelFile(temp);

        var inspect = await service.InspectAsync(path, trustConfirmed: false);
        var predict = await service.PredictAsync(new PythonPredictionRequest(path, false, "auto", null, new[] { (IReadOnlyList<string>)new[] { "1" } }));
        var convert = await service.ConvertAsync(new PythonConversionRequest(path, Path.Combine(temp.Path, "m.onnx"), false));

        Assert.Equal(PythonErrorCode.TrustRequired, inspect.Error!.Code);
        Assert.Equal(PythonErrorCode.TrustRequired, predict.Error!.Code);
        Assert.Equal(PythonErrorCode.TrustRequired, convert.Error!.Code);
        Assert.Null(worker.Command);
    }

    [Fact]
    public async Task ReportsMissingRuntimeAndMissingPackages()
    {
        using var temp = new TempDirectory();
        var path = ModelFile(temp);

        var none = await new PythonModelService(new FakeRuntime(), new FakeWorker()).InspectAsync(path, true);
        Assert.Equal(PythonErrorCode.NoRuntime, none.Error!.Code);

        var partial = new FakeRuntime { Active = Interpreter(PythonSource.System, "numpy", "scikit-learn", "joblib") };
        var service = new PythonModelService(partial, new FakeWorker());
        Assert.True((await service.InspectAsync(path, true)).IsSuccess);

        var convert = await service.ConvertAsync(new PythonConversionRequest(path, Path.Combine(temp.Path, "m.onnx"), true));
        Assert.Equal(PythonErrorCode.MissingPackages, convert.Error!.Code);
        Assert.Contains("skl2onnx", convert.Error.Message);
        Assert.Contains("pip install", convert.Error.Message);
    }

    [Fact]
    public async Task ValidatesFilesAndArguments()
    {
        using var temp = new TempDirectory();
        var service = new PythonModelService(new FakeRuntime { Active = Interpreter(PythonSource.Managed, Everything) }, new FakeWorker());

        Assert.Equal(PythonErrorCode.FileNotFound, (await service.InspectAsync(Path.Combine(temp.Path, "none.joblib"), true)).Error!.Code);
        Assert.Equal(PythonErrorCode.InvalidFormat, (await service.InspectAsync(ModelFile(temp, "model.txt"), true)).Error!.Code);
        Assert.Equal(PythonErrorCode.InvalidInput, (await service.PredictAsync(new PythonPredictionRequest(ModelFile(temp), true, "auto", null, Array.Empty<IReadOnlyList<string>>()))).Error!.Code);
        Assert.Equal(PythonErrorCode.InvalidInput, (await service.ConvertAsync(new PythonConversionRequest(ModelFile(temp), Path.Combine(temp.Path, "m.txt"), true))).Error!.Code);
        Assert.Equal(PythonErrorCode.InvalidInput, (await service.ConvertAsync(new PythonConversionRequest(ModelFile(temp), Path.Combine(temp.Path, "m.onnx"), true, TargetOpset: 0))).Error!.Code);
    }

    [Fact]
    public async Task ParsesInspectionPredictionAndConversionResponses()
    {
        using var temp = new TempDirectory();
        var path = ModelFile(temp);
        var worker = new FakeWorker();
        var service = new PythonModelService(new FakeRuntime { Active = Interpreter(PythonSource.Managed, Everything) }, worker);

        worker.Response = JsonNode.Parse("""
            {"className":"Pipeline","module":"sklearn.pipeline","isPipeline":true,
             "steps":[{"name":"s","className":"StandardScaler"},{"name":"c","className":"LogisticRegression"}],
             "finalEstimator":"LogisticRegression","isClassifier":true,"isTextModel":false,"nFeaturesIn":4,
             "featureNamesIn":null,"classes":[0,1,2],"methods":["predict","predict_proba"],
             "parameters":{"memory":"None"},"trainedWithSklearn":"1.4.0","runtimeSklearn":"1.9.1","warnings":["w"]}
            """)!.AsObject();
        var info = (await service.InspectAsync(path, true)).Value!;
        Assert.Equal("Pipeline(StandardScaler -> LogisticRegression)", info.Summary);
        Assert.Equal(4, info.FeatureCount);
        Assert.Null(info.FeatureNames);
        Assert.Equal(new[] { "0", "1", "2" }, info.Classes);
        Assert.Equal("1.4.0", info.TrainedWithSklearn);
        Assert.Equal("inspect", worker.Command);

        worker.Response = JsonNode.Parse("""
            {"outputs":[{"name":"predict","dtype":"int64","shape":[2],"values":[0,2],"truncated":false},
                        {"name":"predict_proba","dtype":"float64","shape":[2,2],"values":[[0.123456789,0.876543211],[null,1.0]],"truncated":false}],
             "classes":[0,2],"rowCount":2,"warnings":[]}
            """)!.AsObject();
        var prediction = (await service.PredictAsync(new PythonPredictionRequest(path, true, "all", new[] { "a", "b" },
            new[] { (IReadOnlyList<string>)new[] { "1", "2" }, new[] { "3", "4" } }))).Value!;
        Assert.Equal(new[] { "0", "2" }, prediction.Outputs[0].Rows);
        Assert.Equal(new[] { "[0.123457, 0.876543]", "[NaN, 1]" }, prediction.Outputs[1].Rows);
        Assert.Equal("all", worker.Request!["method"]!.GetValue<string>());
        Assert.Equal(2, worker.Request["rows"]!.AsArray().Count);
        Assert.Equal("a", worker.Request["columns"]![0]!.GetValue<string>());

        worker.Response = JsonNode.Parse("""
            {"outputPath":"m.onnx","targetOpset":18,"requestedOpset":18,"size":123,
             "inputs":[{"name":"float_input","elementType":1,"shape":[null,4]}],
             "outputs":[{"name":"label","elementType":7,"shape":[null]}],
             "validation":{"performed":true,"passed":true,"detail":"ok"},"warnings":[]}
            """)!.AsObject();
        var conversion = (await service.ConvertAsync(new PythonConversionRequest(path, Path.Combine(temp.Path, "m.onnx"), true, InputTypes: new[] { "a:float" }))).Value!;
        Assert.Equal("float_input [?, 4]", conversion.Inputs[0].Display);
        Assert.True(conversion.Validation.Passed);
        Assert.False(worker.Request!["zipmap"]!.GetValue<bool>());
        Assert.Equal("a:float", worker.Request["inputTypes"]![0]!.GetValue<string>());
    }
}

public class PythonModelRegistryTests
{
    [Fact]
    public void RegistersOncePerFileAndRaisesEvents()
    {
        using var temp = new TempDirectory();
        var path = Path.Combine(temp.Path, "m.joblib");
        File.WriteAllText(path, "abc");
        var registry = new PythonModelRegistry();
        var added = new List<PythonModel>();
        var removed = new List<PythonModel>();
        registry.ModelAdded += (_, m) => added.Add(m);
        registry.ModelRemoved += (_, m) => removed.Add(m);

        var first = registry.Register(path);
        var second = registry.Register(path);

        Assert.Same(first, second);
        Assert.Single(added);
        Assert.Equal(3, first.FileSize);
        Assert.Equal("m.joblib", first.Name);
        Assert.True(registry.Unload(first.Id));
        Assert.False(registry.Unload(first.Id));
        Assert.Single(removed);
        Assert.Empty(registry.Models);
    }

    [Theory]
    [InlineData("a.joblib", true)]
    [InlineData("a.PKL", true)]
    [InlineData("a.pickle", true)]
    [InlineData("a.onnx", false)]
    [InlineData("a", false)]
    public void RecognisesPythonModelFiles(string name, bool expected)
        => Assert.Equal(expected, PythonModel.IsPythonModelFile(name));
}

public class ProcessRunnerTests
{
    [Fact]
    public async Task CapturesOutputAndExitCode()
    {
        var result = await new ProcessRunner().RunAsync(new ProcessSpec("dotnet", new[] { "--version" }));

        Assert.True(result.Succeeded);
        Assert.Matches(@"^\d+\.\d+", result.StdOut.Trim());
    }

    [Fact]
    public async Task CancellationStopsTheProcess()
    {
        var script = OperatingSystem.IsWindows() ? "ping" : "sleep";
        var args = OperatingSystem.IsWindows() ? new[] { "-n", "30", "127.0.0.1" } : new[] { "30" };
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => new ProcessRunner().RunAsync(new ProcessSpec(script, args), cancellation.Token));
    }

    [Fact]
    public async Task TimeoutIsReported()
    {
        var script = OperatingSystem.IsWindows() ? "ping" : "sleep";
        var args = OperatingSystem.IsWindows() ? new[] { "-n", "30", "127.0.0.1" } : new[] { "30" };

        var result = await new ProcessRunner().RunAsync(new ProcessSpec(script, args, Timeout: TimeSpan.FromMilliseconds(300)));

        Assert.True(result.TimedOut);
        Assert.False(result.Succeeded);
    }
}
