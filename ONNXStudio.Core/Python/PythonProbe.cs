using System.Text.Json.Nodes;

namespace ONNXStudio.Core.Python;

/// <summary>
/// Inspects a Python interpreter: version, architecture and installed packages.
/// </summary>
public interface IPythonProbe
{
    /// <summary>Returns null when the executable is not a usable Python (>= 3.9).</summary>
    Task<PythonInterpreter?> ProbeAsync(string executablePath, PythonSource source, CancellationToken cancellationToken = default);
}

public sealed class PythonProbe : IPythonProbe
{
    private const string Script = """
        import sys, json, platform
        try:
            from importlib import metadata
        except ImportError:
            metadata = None
        names = __NAMES__
        packages = {}
        for name in names:
            try:
                packages[name] = metadata.version(name) if metadata else None
            except Exception:
                packages[name] = None
        print(json.dumps({"version": platform.python_version(), "arch": platform.machine(), "packages": packages}))
        """;

    private readonly IProcessRunner _runner;

    public PythonProbe(IProcessRunner runner) => _runner = runner;

    public async Task<PythonInterpreter?> ProbeAsync(string executablePath, PythonSource source, CancellationToken cancellationToken = default)
    {
        if (!File.Exists(executablePath)) return null;

        var names = "[" + string.Join(", ", PythonRequirements.Probed.Select(n => "\"" + n + "\"")) + "]";
        ProcessResult result;
        try
        {
            result = await _runner.RunAsync(new ProcessSpec(
                executablePath,
                new[] { "-c", Script.Replace("__NAMES__", names) },
                Environment: PythonEnvironment.For(source),
                Timeout: TimeSpan.FromSeconds(30)), cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or IOException or InvalidOperationException)
        {
            return null;
        }

        if (!result.Succeeded) return null;

        // Site customisations may print before our JSON: the last line is ours.
        var line = result.StdOut.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).LastOrDefault();
        if (line == null) return null;

        try
        {
            if (JsonNode.Parse(line) is not JsonObject json) return null;
            var version = json["version"]?.GetValue<string>();
            if (version == null || !IsSupportedVersion(version)) return null;

            var packages = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
            if (json["packages"] is JsonObject map)
            {
                foreach (var (name, value) in map) packages[name] = value?.GetValue<string>();
            }
            return new PythonInterpreter(executablePath, version, json["arch"]?.GetValue<string>() ?? "", source, packages);
        }
        catch (Exception ex) when (ex is System.Text.Json.JsonException or InvalidOperationException or FormatException)
        {
            return null;
        }
    }

    public static bool IsSupportedVersion(string version)
    {
        var minimum = Version.Parse(PythonRequirements.MinimumPythonVersion);
        var parts = version.Split('.');
        return parts.Length >= 2
               && int.TryParse(parts[0], out var major)
               && int.TryParse(new string(parts[1].TakeWhile(char.IsDigit).ToArray()), out var minor)
               && new Version(major, minor) >= minimum;
    }
}

/// <summary>Environment variables applied to every Python process.</summary>
public static class PythonEnvironment
{
    public static IReadOnlyDictionary<string, string?> For(PythonSource source)
    {
        var env = new Dictionary<string, string?>
        {
            ["PYTHONIOENCODING"] = "utf-8",
            ["PYTHONUTF8"] = "1",
            ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1",
            // A managed runtime must not see packages from the user site directory.
            ["PYTHONNOUSERSITE"] = source == PythonSource.Managed ? "1" : null,
            ["PYTHONHOME"] = null,
        };
        if (source == PythonSource.Managed) env["PYTHONPATH"] = null;
        return env;
    }
}
