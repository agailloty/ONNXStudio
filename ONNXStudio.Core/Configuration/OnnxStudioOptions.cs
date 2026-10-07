namespace ONNXStudio.Core.Configuration;

/// <summary>
/// Application-wide settings (from configuration or defaults).
/// </summary>
public sealed class OnnxStudioOptions
{
    public const string SectionName = "ONNXStudio";

    public int MaxModelSizeMB { get; set; } = 2048;
    public int MaxConcurrentInferences { get; set; } = 4;
    public int SessionCacheSize { get; set; } = 5;
    public string SupportedOpsetMax { get; set; } = "18";

    public OnnxStudioApiOptions Api { get; set; } = new();
    public OnnxStudioPythonOptions Python { get; set; } = new();
}

/// <summary>
/// Python support (joblib/pickle inference and ONNX conversion). Nothing Python
/// is shipped with the application: the runtime is installed on demand.
/// </summary>
public sealed class OnnxStudioPythonOptions
{
    /// <summary>Directory holding the managed runtime, worker script and selection. Defaults to %LOCALAPPDATA%/ONNXStudio/python.</summary>
    public string? Directory { get; set; }

    /// <summary>Python minor version downloaded for the managed runtime.</summary>
    public string ManagedVersion { get; set; } = "3.12";

    /// <summary>Fallback python-build-standalone release used when the latest release cannot be queried.</summary>
    public string FallbackReleaseTag { get; set; } = "20261003";

    public string FallbackPythonPatch { get; set; } = "3.12.15";

    public int WorkerTimeoutSeconds { get; set; } = 300;

    public string ResolveDirectory() => string.IsNullOrWhiteSpace(Directory)
        ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ONNXStudio", "python")
        : Directory;
}

public sealed class OnnxStudioApiOptions
{
    public int Port { get; set; } = 5000;
    public int MaxRequestSizeMB { get; set; } = 10;
    public bool EnableSwagger { get; set; } = true;
    public string CorsPolicy { get; set; } = "AllowAllForDevelopment";
}
