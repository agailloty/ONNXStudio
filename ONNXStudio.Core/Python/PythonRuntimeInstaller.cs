using System.Formats.Tar;
using System.IO.Compression;
using System.Security.Cryptography;
using Microsoft.Extensions.Logging;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Python;

/// <summary>
/// Installs the managed Python runtime (download, checksum, extraction) and its packages.
/// </summary>
public interface IPythonRuntimeInstaller
{
    /// <summary>Installs (or reinstalls) the managed Python; returns the interpreter executable.</summary>
    Task<Result<string, PythonError>> InstallPythonAsync(IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default);

    /// <summary>Installs packages into the given managed interpreter with pip (binary wheels only).</summary>
    Task<Result<bool, PythonError>> InstallPackagesAsync(string executablePath, IReadOnlyList<string> packages,
        IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default);

    /// <summary>Deletes the managed runtime.</summary>
    void RemoveManagedRuntime();
}

public sealed class PythonRuntimeInstaller : IPythonRuntimeInstaller
{
    private readonly PythonPaths _paths;
    private readonly PythonDistributionResolver _resolver;
    private readonly IPythonDownloader _downloader;
    private readonly IProcessRunner _runner;
    private readonly ILogger<PythonRuntimeInstaller> _logger;

    public PythonRuntimeInstaller(
        PythonPaths paths,
        PythonDistributionResolver resolver,
        IPythonDownloader downloader,
        IProcessRunner runner,
        ILogger<PythonRuntimeInstaller> logger)
    {
        _paths = paths;
        _resolver = resolver;
        _downloader = downloader;
        _runner = runner;
        _logger = logger;
    }

    public async Task<Result<string, PythonError>> InstallPythonAsync(IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default)
    {
        progress?.Report(new PythonInstallProgress("resolve", "Looking for a Python build..."));
        var resolved = await _resolver.ResolveAsync(cancellationToken).ConfigureAwait(false);
        if (resolved.IsFailure) return Result<string, PythonError>.Failure(resolved.Error!);
        var distribution = resolved.Value!;

        var archive = Path.Combine(_paths.DownloadDirectory, distribution.FileName);
        var staging = _paths.ManagedRoot + ".staging";
        try
        {
            progress?.Report(new PythonInstallProgress("download", $"Downloading Python {distribution.Version}...", 0));
            var downloadProgress = new Progress<double>(f =>
                progress?.Report(new PythonInstallProgress("download", $"Downloading Python {distribution.Version}... {f:P0}", f)));
            try
            {
                await _downloader.DownloadFileAsync(distribution.Url.ToString(), archive, downloadProgress, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is HttpRequestException or IOException)
            {
                _logger.LogError(ex, "Python download failed");
                return Fail(PythonErrorCode.DownloadFailed,
                    "Python could not be downloaded. Check your internet connection and free disk space.", ex.Message);
            }

            progress?.Report(new PythonInstallProgress("verify", "Verifying download..."));
            var actual = await ComputeSha256Async(archive, cancellationToken).ConfigureAwait(false);
            if (!string.Equals(actual, distribution.Sha256, StringComparison.OrdinalIgnoreCase))
            {
                return Fail(PythonErrorCode.ChecksumMismatch,
                    "The downloaded Python archive is corrupted or was tampered with (checksum mismatch). Nothing was installed.",
                    $"expected {distribution.Sha256}, got {actual}");
            }

            progress?.Report(new PythonInstallProgress("extract", "Extracting Python..."));
            if (Directory.Exists(staging)) Directory.Delete(staging, recursive: true);
            await Task.Run(() => Extract(archive, staging), cancellationToken).ConfigureAwait(false);

            var stagedExecutable = Path.Combine(staging, Path.GetRelativePath(_paths.ManagedRoot, _paths.ManagedExecutable));
            if (!File.Exists(stagedExecutable))
            {
                return Fail(PythonErrorCode.InstallFailed, "The Python archive did not contain the expected interpreter.");
            }

            if (Directory.Exists(_paths.ManagedRoot)) Directory.Delete(_paths.ManagedRoot, recursive: true);
            Directory.Move(staging, _paths.ManagedRoot);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidDataException)
        {
            _logger.LogError(ex, "Python installation failed");
            return Fail(PythonErrorCode.InstallFailed,
                "Python could not be installed. Check folder permissions and free disk space.", ex.Message);
        }
        finally
        {
            TryDelete(archive);
            TryDeleteDirectory(staging);
        }

        // python-build-standalone ships pip; repair it if it is somehow missing.
        var pip = await RunPipAsync(_paths.ManagedExecutable, new[] { "-m", "pip", "--version" }, null, cancellationToken).ConfigureAwait(false);
        if (!pip.Succeeded)
        {
            await RunPipAsync(_paths.ManagedExecutable, new[] { "-m", "ensurepip", "--upgrade" }, null, cancellationToken).ConfigureAwait(false);
        }

        progress?.Report(new PythonInstallProgress("done", $"Python {distribution.Version} installed.", 1));
        return Result<string, PythonError>.Success(_paths.ManagedExecutable);
    }

    public async Task<Result<bool, PythonError>> InstallPackagesAsync(
        string executablePath,
        IReadOnlyList<string> packages,
        IProgress<PythonInstallProgress>? progress,
        CancellationToken cancellationToken = default)
    {
        if (packages.Count == 0) return Result<bool, PythonError>.Success(true);

        progress?.Report(new PythonInstallProgress("pip", $"Installing {string.Join(", ", packages)}..."));
        var arguments = new List<string> { "-m", "pip", "install", "--upgrade", "--only-binary", ":all:", "--no-input" };
        arguments.AddRange(packages);

        ProcessResult result;
        try
        {
            result = await RunPipAsync(executablePath, arguments, line => progress?.Report(new PythonInstallProgress("pip", line)), cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or IOException)
        {
            return Result<bool, PythonError>.Failure(new PythonError(PythonErrorCode.InstallFailed,
                "pip could not be started.", ex.Message));
        }

        if (!result.Succeeded)
        {
            _logger.LogError("pip install failed: {Output}", result.StdErr);
            return Result<bool, PythonError>.Failure(new PythonError(PythonErrorCode.InstallFailed,
                result.TimedOut
                    ? "Package installation timed out."
                    : "Packages could not be installed. Check your internet connection (details in the log above).",
                Tail(result.StdErr, 2000)));
        }

        progress?.Report(new PythonInstallProgress("done", "Packages installed.", 1));
        return Result<bool, PythonError>.Success(true);
    }

    public void RemoveManagedRuntime() => TryDeleteDirectory(_paths.ManagedRoot);

    private Task<ProcessResult> RunPipAsync(string executable, IReadOnlyList<string> arguments, Action<string>? onLine, CancellationToken cancellationToken)
        => _runner.RunAsync(new ProcessSpec(
            executable,
            arguments,
            Environment: PythonEnvironment.For(PythonSource.Managed),
            Timeout: TimeSpan.FromMinutes(20),
            OnOutputLine: onLine), cancellationToken);

    private static void Extract(string archive, string destination)
    {
        Directory.CreateDirectory(destination);
        using var file = File.OpenRead(archive);
        using var gzip = new GZipStream(file, CompressionMode.Decompress);
        TarFile.ExtractToDirectory(gzip, destination, overwriteFiles: true);
    }

    private static async Task<string> ComputeSha256Async(string path, CancellationToken cancellationToken)
    {
        await using var stream = File.OpenRead(path);
        return Convert.ToHexString(await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false)).ToLowerInvariant();
    }

    private static string Tail(string text, int max) => text.Length <= max ? text : text[^max..];

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); } catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    private static void TryDeleteDirectory(string path)
    {
        try { if (Directory.Exists(path)) Directory.Delete(path, recursive: true); } catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    private static Result<string, PythonError> Fail(PythonErrorCode code, string message, string details = "")
        => Result<string, PythonError>.Failure(new PythonError(code, message, details));
}
