namespace ONNXStudio.Core.Python;

/// <summary>Where a Python interpreter comes from.</summary>
public enum PythonSource
{
    /// <summary>Downloaded and installed by ONNX Studio in its own directory.</summary>
    Managed,

    /// <summary>Found automatically (py launcher, PATH, active venv / conda env...).</summary>
    System,

    /// <summary>Explicitly provided by the user (interpreter file or environment folder).</summary>
    Custom
}

/// <summary>Python packages ONNX Studio relies on.</summary>
public static class PythonRequirements
{
    public const string MinimumPythonVersion = "3.9";

    /// <summary>Installed in the managed runtime (pip distribution names).</summary>
    public static readonly IReadOnlyList<string> Managed = new[]
    {
        "numpy", "scipy", "pandas", "scikit-learn", "joblib", "skl2onnx", "onnx", "onnxruntime"
    };

    /// <summary>Needed to load a joblib/pickle model and run inference.</summary>
    public static readonly IReadOnlyList<string> Inference = new[] { "numpy", "scikit-learn", "joblib" };

    /// <summary>Needed on top of <see cref="Inference"/> to convert a model to ONNX.</summary>
    public static readonly IReadOnlyList<string> Conversion = new[] { "skl2onnx", "onnx" };

    /// <summary>Reported when present (never required).</summary>
    public static readonly IReadOnlyList<string> Optional = new[] { "onnxmltools", "xgboost", "lightgbm" };

    public static IEnumerable<string> Probed => Managed.Concat(Optional).Distinct(StringComparer.OrdinalIgnoreCase);
}

/// <summary>A probed Python interpreter and the packages found in it.</summary>
public sealed class PythonInterpreter
{
    public string ExecutablePath { get; }
    public string Version { get; }
    public string Architecture { get; }
    public PythonSource Source { get; }

    /// <summary>Package name to installed version (null when not installed).</summary>
    public IReadOnlyDictionary<string, string?> Packages { get; }

    public PythonInterpreter(string executablePath, string version, string architecture, PythonSource source,
        IReadOnlyDictionary<string, string?> packages)
    {
        ExecutablePath = executablePath;
        Version = version;
        Architecture = architecture;
        Source = source;
        Packages = packages;
    }

    public bool HasPackage(string name) => Packages.TryGetValue(name, out var v) && v != null;

    public string? PackageVersion(string name) => Packages.TryGetValue(name, out var v) ? v : null;

    public IReadOnlyList<string> MissingForInference => PythonRequirements.Inference.Where(p => !HasPackage(p)).ToArray();

    public IReadOnlyList<string> MissingForConversion =>
        PythonRequirements.Inference.Concat(PythonRequirements.Conversion).Where(p => !HasPackage(p)).ToArray();

    /// <summary>Packages of the managed set that are not installed.</summary>
    public IReadOnlyList<string> MissingManaged => PythonRequirements.Managed.Where(p => !HasPackage(p)).ToArray();

    public bool CanRunInference => MissingForInference.Count == 0;
    public bool CanConvert => MissingForConversion.Count == 0;

    public string DisplayName => $"Python {Version} ({Source}) - {ExecutablePath}";

    public override string ToString() => DisplayName;
}

public enum PythonErrorCode
{
    None,
    NoRuntime,
    MissingPackages,
    InvalidInterpreter,
    UnsupportedPlatform,
    DownloadFailed,
    ChecksumMismatch,
    InstallFailed,
    TrustRequired,
    FileNotFound,
    InvalidInput,
    InvalidFormat,
    LoadFailed,
    MissingModule,
    PredictionFailed,
    UnsupportedModel,
    ConversionFailed,
    WorkerFailed,
    Timeout,
    Cancelled
}

/// <summary>User-friendly Python error with technical details reserved for logs.</summary>
public sealed class PythonError
{
    public PythonErrorCode Code { get; }
    public string Message { get; }
    public string TechnicalDetails { get; }

    public PythonError(PythonErrorCode code, string message, string technicalDetails = "")
    {
        Code = code;
        Message = message;
        TechnicalDetails = technicalDetails;
    }

    public override string ToString() => $"[{Code}] {Message}";
}

/// <summary>Progress notification of a long running installation step.</summary>
public sealed record PythonInstallProgress(string Stage, string Message, double? Percent = null);

/// <summary>Persisted choice of the interpreter ONNX Studio runs Python models with.</summary>
public sealed record PythonSelection(string? InterpreterPath, PythonSource Source = PythonSource.Custom)
{
    /// <summary>No explicit choice: the managed runtime, else the first system interpreter with all packages.</summary>
    public static PythonSelection Automatic { get; } = new PythonSelection(InterpreterPath: null);

    public bool IsAutomatic => string.IsNullOrWhiteSpace(InterpreterPath);
}
