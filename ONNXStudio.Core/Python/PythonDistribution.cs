using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Python;

/// <summary>A downloadable standalone CPython build with its expected checksum.</summary>
public sealed record PythonDistribution(string Version, string FileName, Uri Url, string Sha256);

/// <summary>Network access used to install Python (abstracted for tests).</summary>
public interface IPythonDownloader
{
    Task<string> GetStringAsync(string url, CancellationToken cancellationToken = default);

    /// <summary>Downloads to <paramref name="destination"/>; progress is a 0..1 fraction (when the size is known).</summary>
    Task DownloadFileAsync(string url, string destination, IProgress<double>? progress, CancellationToken cancellationToken = default);
}

public sealed class PythonDownloader : IPythonDownloader, IDisposable
{
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromMinutes(30) };

    public PythonDownloader()
    {
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("ONNXStudio");
    }

    public Task<string> GetStringAsync(string url, CancellationToken cancellationToken = default)
        => _http.GetStringAsync(url, cancellationToken);

    public async Task DownloadFileAsync(string url, string destination, IProgress<double>? progress, CancellationToken cancellationToken = default)
    {
        using var response = await _http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        var total = response.Content.Headers.ContentLength;

        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        await using var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        await using var target = File.Create(destination);

        var buffer = new byte[81920];
        long done = 0;
        int read;
        while ((read = await source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
        {
            await target.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
            done += read;
            if (total is > 0) progress?.Report((double)done / total.Value);
        }
    }

    public void Dispose() => _http.Dispose();
}

/// <summary>
/// Picks the standalone CPython build (astral-sh/python-build-standalone) matching
/// this platform, with the SHA-256 published in the release.
/// </summary>
public sealed class PythonDistributionResolver
{
    private const string LatestReleaseUrl = "https://raw.githubusercontent.com/astral-sh/python-build-standalone/latest-release/latest-release.json";
    private const string DownloadBase = "https://github.com/astral-sh/python-build-standalone/releases/download/";

    private readonly IPythonDownloader _downloader;
    private readonly OnnxStudioPythonOptions _options;

    public PythonDistributionResolver(IPythonDownloader downloader, OnnxStudioPythonOptions options)
    {
        _downloader = downloader;
        _options = options;
    }

    /// <summary>Platform triple used in python-build-standalone asset names, or null when unsupported.</summary>
    public static string? CurrentTriple()
    {
        var arm = RuntimeInformation.OSArchitecture == Architecture.Arm64;
        if (!arm && RuntimeInformation.OSArchitecture != Architecture.X64) return null;
        if (OperatingSystem.IsWindows()) return arm ? "aarch64-pc-windows-msvc" : "x86_64-pc-windows-msvc";
        if (OperatingSystem.IsMacOS()) return arm ? "aarch64-apple-darwin" : "x86_64-apple-darwin";
        if (OperatingSystem.IsLinux()) return arm ? "aarch64-unknown-linux-gnu" : "x86_64-unknown-linux-gnu";
        return null;
    }

    public async Task<Result<PythonDistribution, PythonError>> ResolveAsync(CancellationToken cancellationToken = default)
    {
        var triple = CurrentTriple();
        if (triple == null)
        {
            return Fail(PythonErrorCode.UnsupportedPlatform,
                "Automatic Python installation is not available on this platform. Install Python yourself and select it in the settings.");
        }

        var tag = _options.FallbackReleaseTag;
        try
        {
            var latest = JsonNode.Parse(await _downloader.GetStringAsync(LatestReleaseUrl, cancellationToken).ConfigureAwait(false));
            if (latest?["tag"]?.GetValue<string>() is { Length: > 0 } latestTag) tag = latestTag;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or System.Text.Json.JsonException or InvalidOperationException)
        {
            if (cancellationToken.IsCancellationRequested) throw;
            // Fall back to the pinned release below.
        }

        string sums;
        try
        {
            sums = await _downloader.GetStringAsync($"{DownloadBase}{tag}/SHA256SUMS", cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            if (cancellationToken.IsCancellationRequested) throw;
            return Fail(PythonErrorCode.DownloadFailed,
                "Could not reach GitHub to find a Python build. Check your internet connection or select an existing Python in the settings.",
                ex.Message);
        }

        var distribution = Select(sums, tag, triple, _options.ManagedVersion);
        return distribution == null
            ? Fail(PythonErrorCode.DownloadFailed,
                $"No Python {_options.ManagedVersion} build was found for {triple} in release {tag}.")
            : Result<PythonDistribution, PythonError>.Success(distribution);
    }

    /// <summary>Selects the newest matching "install_only_stripped" build listed in a SHA256SUMS file.</summary>
    public static PythonDistribution? Select(string sha256Sums, string tag, string triple, string minorVersion)
    {
        var pattern = new Regex(
            @"^(?<hash>[0-9a-fA-F]{64})\s+\*?(?<file>cpython-(?<ver>" + Regex.Escape(minorVersion) + @"\.\d+)\+" + Regex.Escape(tag) + "-" + Regex.Escape(triple) + @"-install_only_stripped\.tar\.gz)\s*$",
            RegexOptions.Multiline);

        PythonDistribution? best = null;
        Version? bestVersion = null;
        foreach (Match match in pattern.Matches(sha256Sums))
        {
            var version = Version.Parse(match.Groups["ver"].Value);
            if (bestVersion != null && version <= bestVersion) continue;
            bestVersion = version;
            var file = match.Groups["file"].Value;
            best = new PythonDistribution(
                match.Groups["ver"].Value,
                file,
                new Uri($"{DownloadBase}{tag}/{Uri.EscapeDataString(file)}"),
                match.Groups["hash"].Value.ToLowerInvariant());
        }
        return best;
    }

    private static Result<PythonDistribution, PythonError> Fail(PythonErrorCode code, string message, string details = "")
        => Result<PythonDistribution, PythonError>.Failure(new PythonError(code, message, details));
}
