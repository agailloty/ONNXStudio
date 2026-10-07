using System.Text.RegularExpressions;

namespace ONNXStudio.Core.Python;

/// <summary>
/// Finds Python interpreters already present on the machine (py launcher, PATH,
/// active virtual / conda environment) and resolves user supplied locations.
/// </summary>
public interface IPythonDiscovery
{
    /// <summary>Candidate interpreter executables, deduplicated, not yet probed.</summary>
    Task<IReadOnlyList<string>> FindCandidatesAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Accepts an interpreter file or an environment / installation folder
    /// (venv, conda env, Python home) and returns the interpreter executable, or null.
    /// </summary>
    string? ResolveExecutable(string pathOrFolder);
}

public sealed partial class PythonDiscovery : IPythonDiscovery
{
    private readonly IProcessRunner _runner;

    public PythonDiscovery(IProcessRunner runner) => _runner = runner;

    public async Task<IReadOnlyList<string>> FindCandidatesAsync(CancellationToken cancellationToken = default)
    {
        var found = new List<string>();

        foreach (var variable in new[] { "VIRTUAL_ENV", "CONDA_PREFIX" })
        {
            if (Environment.GetEnvironmentVariable(variable) is { Length: > 0 } prefix && ResolveExecutable(prefix) is { } exe)
                found.Add(exe);
        }

        if (OperatingSystem.IsWindows())
        {
            found.AddRange(await FindWithPyLauncherAsync(cancellationToken).ConfigureAwait(false));
        }

        var names = OperatingSystem.IsWindows() ? new[] { "python.exe", "python3.exe" } : new[] { "python3", "python" };
        var directories = (Environment.GetEnvironmentVariable("PATH") ?? string.Empty)
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (!OperatingSystem.IsWindows())
        {
            directories = directories.Concat(new[] { "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin" }).ToArray();
        }

        foreach (var directory in directories)
        {
            foreach (var name in names)
            {
                var candidate = Path.Combine(directory, name);
                if (File.Exists(candidate)) found.Add(candidate);
            }
        }

        return found
            .Where(path => !IsWindowsStoreStub(path))
            .Select(NormalizePath)
            .Distinct(PathComparer)
            .ToArray();
    }

    public string? ResolveExecutable(string pathOrFolder)
    {
        if (string.IsNullOrWhiteSpace(pathOrFolder)) return null;
        var path = pathOrFolder.Trim().Trim('"');

        if (File.Exists(path)) return NormalizePath(path);
        if (!Directory.Exists(path)) return null;

        var relative = OperatingSystem.IsWindows()
            ? new[] { "python.exe", Path.Combine("Scripts", "python.exe"), Path.Combine("python", "python.exe") }
            : new[] { Path.Combine("bin", "python3"), Path.Combine("bin", "python"), Path.Combine("python", "bin", "python3") };

        foreach (var item in relative)
        {
            var candidate = Path.Combine(path, item);
            if (File.Exists(candidate)) return NormalizePath(candidate);
        }
        return null;
    }

    private async Task<IEnumerable<string>> FindWithPyLauncherAsync(CancellationToken cancellationToken)
    {
        try
        {
            var result = await _runner.RunAsync(new ProcessSpec("py", new[] { "-0p" }, Timeout: TimeSpan.FromSeconds(10)), cancellationToken)
                .ConfigureAwait(false);
            return result.Succeeded ? ParsePyLauncherOutput(result.StdOut) : Array.Empty<string>();
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or IOException or InvalidOperationException)
        {
            return Array.Empty<string>();
        }
    }

    /// <summary>Parses the output of "py -0p" (one "-V:3.12 *  C:\...\python.exe" per line).</summary>
    public static IEnumerable<string> ParsePyLauncherOutput(string output)
    {
        foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var match = PyLauncherLine().Match(line);
            if (match.Success) yield return match.Groups["path"].Value.Trim();
        }
    }

    [GeneratedRegex(@"^\s*-\S+\s+(?:\*\s+)?(?<path>(?:[A-Za-z]:\\|/).*?python\w*(?:\.exe)?)\s*$", RegexOptions.IgnoreCase)]
    private static partial Regex PyLauncherLine();

    // "python.exe" aliases in WindowsApps open the Microsoft Store instead of running Python.
    private static bool IsWindowsStoreStub(string path)
        => OperatingSystem.IsWindows() && path.Contains(@"\Microsoft\WindowsApps\", StringComparison.OrdinalIgnoreCase);

    private static string NormalizePath(string path) => Path.GetFullPath(path);

    private static StringComparer PathComparer
        => OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
}
