using System.Text.Json.Nodes;
using Microsoft.Extensions.Logging;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Python;

/// <summary>
/// Knows which Python interpreter ONNX Studio uses: the managed runtime it
/// installs itself, an interpreter found on the machine, or one the user points to.
/// </summary>
public interface IPythonRuntimeService
{
    /// <summary>Raised when the selection or the content of an interpreter changed.</summary>
    event Action? Changed;

    PythonSelection Selection { get; }

    void Select(PythonSelection selection);

    bool IsManagedInstalled { get; }

    /// <summary>The managed interpreter (if installed) followed by the interpreters found on the machine.</summary>
    Task<IReadOnlyList<PythonInterpreter>> DiscoverAsync(CancellationToken cancellationToken = default);

    /// <summary>Probes an interpreter file or environment folder chosen by the user.</summary>
    Task<Result<PythonInterpreter, PythonError>> ProbeCustomAsync(string pathOrFolder, CancellationToken cancellationToken = default);

    /// <summary>The interpreter currently in use, or null when none is usable.</summary>
    Task<PythonInterpreter?> GetActiveAsync(bool refresh = false, CancellationToken cancellationToken = default);

    Task<Result<PythonInterpreter, PythonError>> InstallPythonAsync(IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default);

    /// <summary>Installs the missing packages into the managed runtime (never touches other interpreters).</summary>
    Task<Result<PythonInterpreter, PythonError>> InstallPackagesAsync(IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default);

    Task RemoveManagedAsync(CancellationToken cancellationToken = default);
}

public sealed class PythonRuntimeService : IPythonRuntimeService
{
    private readonly PythonPaths _paths;
    private readonly IPythonDiscovery _discovery;
    private readonly IPythonProbe _probe;
    private readonly IPythonRuntimeInstaller _installer;
    private readonly ILogger<PythonRuntimeService> _logger;
    private readonly SemaphoreSlim _installGate = new(1, 1);
    private readonly object _selectionLock = new();
    private PythonSelection _selection;
    private PythonInterpreter? _active;
    private bool _activeResolved;

    public event Action? Changed;

    public PythonRuntimeService(
        PythonPaths paths,
        IPythonDiscovery discovery,
        IPythonProbe probe,
        IPythonRuntimeInstaller installer,
        ILogger<PythonRuntimeService> logger)
    {
        _paths = paths;
        _discovery = discovery;
        _probe = probe;
        _installer = installer;
        _logger = logger;
        _selection = LoadSelection();
    }

    public PythonSelection Selection
    {
        get { lock (_selectionLock) return _selection; }
    }

    public bool IsManagedInstalled => File.Exists(_paths.ManagedExecutable);

    public void Select(PythonSelection selection)
    {
        lock (_selectionLock)
        {
            _selection = selection;
            _active = null;
            _activeResolved = false;
        }
        SaveSelection(selection);
        Changed?.Invoke();
    }

    public async Task<IReadOnlyList<PythonInterpreter>> DiscoverAsync(CancellationToken cancellationToken = default)
    {
        var result = new List<PythonInterpreter>();
        var known = new HashSet<string>(PathComparer);

        if (IsManagedInstalled &&
            await _probe.ProbeAsync(_paths.ManagedExecutable, PythonSource.Managed, cancellationToken).ConfigureAwait(false) is { } managed)
        {
            result.Add(managed);
            known.Add(managed.ExecutablePath);
        }

        var candidates = (await _discovery.FindCandidatesAsync(cancellationToken).ConfigureAwait(false))
            .Where(known.Add)
            .ToArray();
        var probed = await Task.WhenAll(candidates.Select(c => _probe.ProbeAsync(c, PythonSource.System, cancellationToken))).ConfigureAwait(false);
        result.AddRange(probed.OfType<PythonInterpreter>());
        return result;
    }

    public async Task<Result<PythonInterpreter, PythonError>> ProbeCustomAsync(string pathOrFolder, CancellationToken cancellationToken = default)
    {
        var executable = _discovery.ResolveExecutable(pathOrFolder);
        if (executable == null)
        {
            return Result<PythonInterpreter, PythonError>.Failure(new PythonError(PythonErrorCode.InvalidInterpreter,
                "No Python interpreter was found at this location. Select python.exe / python3, or the folder of a virtual environment.",
                pathOrFolder));
        }

        var interpreter = await _probe.ProbeAsync(executable, PythonSource.Custom, cancellationToken).ConfigureAwait(false);
        return interpreter == null
            ? Result<PythonInterpreter, PythonError>.Failure(new PythonError(PythonErrorCode.InvalidInterpreter,
                $"'{executable}' could not be run as Python {PythonRequirements.MinimumPythonVersion} or newer.", executable))
            : Result<PythonInterpreter, PythonError>.Success(interpreter);
    }

    public async Task<PythonInterpreter?> GetActiveAsync(bool refresh = false, CancellationToken cancellationToken = default)
    {
        PythonSelection selection;
        lock (_selectionLock)
        {
            if (_activeResolved && !refresh) return _active;
            selection = _selection;
        }

        var resolved = await ResolveAsync(selection, cancellationToken).ConfigureAwait(false);

        lock (_selectionLock)
        {
            // Ignore the result if the selection changed while probing.
            if (ReferenceEquals(selection, _selection))
            {
                _active = resolved;
                _activeResolved = true;
            }
        }
        return resolved;
    }

    private async Task<PythonInterpreter?> ResolveAsync(PythonSelection selection, CancellationToken cancellationToken)
    {
        if (!selection.IsAutomatic)
        {
            var path = selection.InterpreterPath!;
            var source = PathComparer.Equals(Path.GetFullPath(path), _paths.ManagedExecutable) ? PythonSource.Managed : selection.Source;
            return await _probe.ProbeAsync(path, source, cancellationToken).ConfigureAwait(false);
        }

        var all = await DiscoverAsync(cancellationToken).ConfigureAwait(false);
        return all.FirstOrDefault(i => i.Source == PythonSource.Managed)
               ?? all.FirstOrDefault(i => i.CanRunInference);
    }

    public async Task<Result<PythonInterpreter, PythonError>> InstallPythonAsync(
        IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default)
    {
        await _installGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var installed = await _installer.InstallPythonAsync(progress, cancellationToken).ConfigureAwait(false);
            if (installed.IsFailure) return Result<PythonInterpreter, PythonError>.Failure(installed.Error!);

            // Selecting nothing explicitly keeps the managed runtime as the automatic choice.
            Select(PythonSelection.Automatic);
            var interpreter = await _probe.ProbeAsync(installed.Value!, PythonSource.Managed, cancellationToken).ConfigureAwait(false);
            return interpreter == null
                ? Result<PythonInterpreter, PythonError>.Failure(new PythonError(PythonErrorCode.InstallFailed,
                    "Python was installed but could not be started.", installed.Value!))
                : Result<PythonInterpreter, PythonError>.Success(interpreter);
        }
        finally
        {
            _installGate.Release();
        }
    }

    public async Task<Result<PythonInterpreter, PythonError>> InstallPackagesAsync(
        IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default)
    {
        await _installGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (!IsManagedInstalled)
            {
                return Result<PythonInterpreter, PythonError>.Failure(new PythonError(PythonErrorCode.NoRuntime,
                    "Install the ONNX Studio Python runtime first. Packages are never installed into other interpreters."));
            }

            var current = await _probe.ProbeAsync(_paths.ManagedExecutable, PythonSource.Managed, cancellationToken).ConfigureAwait(false);
            var missing = current?.MissingManaged ?? PythonRequirements.Managed;
            // Packages already present are refreshed too so the set stays consistent.
            var installed = await _installer.InstallPackagesAsync(_paths.ManagedExecutable, PythonRequirements.Managed, progress, cancellationToken).ConfigureAwait(false);
            if (installed.IsFailure) return Result<PythonInterpreter, PythonError>.Failure(installed.Error!);

            _logger.LogInformation("Installed Python packages (previously missing: {Missing})", string.Join(", ", missing));
            var interpreter = await _probe.ProbeAsync(_paths.ManagedExecutable, PythonSource.Managed, cancellationToken).ConfigureAwait(false);
            Select(Selection);
            return interpreter == null
                ? Result<PythonInterpreter, PythonError>.Failure(new PythonError(PythonErrorCode.InstallFailed, "The managed Python could not be started."))
                : Result<PythonInterpreter, PythonError>.Success(interpreter);
        }
        finally
        {
            _installGate.Release();
        }
    }

    public async Task RemoveManagedAsync(CancellationToken cancellationToken = default)
    {
        await _installGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            _installer.RemoveManagedRuntime();
            if (PathComparer.Equals(Selection.InterpreterPath ?? string.Empty, _paths.ManagedExecutable)) Select(PythonSelection.Automatic);
            else Select(Selection);
        }
        finally
        {
            _installGate.Release();
        }
    }

    private PythonSelection LoadSelection()
    {
        try
        {
            if (File.Exists(_paths.SelectionFile) && JsonNode.Parse(File.ReadAllText(_paths.SelectionFile)) is JsonObject json
                && json["interpreter"]?.GetValue<string>() is { Length: > 0 } path)
            {
                var source = Enum.TryParse<PythonSource>(json["source"]?.GetValue<string>(), out var parsed) ? parsed : PythonSource.Custom;
                return new PythonSelection(path, source);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.Text.Json.JsonException or InvalidOperationException or FormatException)
        {
            _logger.LogWarning(ex, "Python selection file ignored");
        }
        return PythonSelection.Automatic;
    }

    private void SaveSelection(PythonSelection selection)
    {
        try
        {
            Directory.CreateDirectory(_paths.Root);
            var temp = _paths.SelectionFile + ".tmp";
            File.WriteAllText(temp, new JsonObject
            {
                ["interpreter"] = selection.InterpreterPath,
                ["source"] = selection.Source.ToString()
            }.ToJsonString());
            File.Move(temp, _paths.SelectionFile, overwrite: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            _logger.LogWarning(ex, "Python selection could not be saved");
        }
    }

    private static StringComparer PathComparer
        => OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
}
