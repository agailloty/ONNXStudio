namespace ONNXStudio.Core.Python;

/// <summary>
/// Paths of everything ONNX Studio keeps for Python support.
/// </summary>
public sealed class PythonPaths
{
    public string Root { get; }

    /// <summary>Directory the managed distribution is extracted to (contains "python").</summary>
    public string ManagedRoot => Path.Combine(Root, "runtime");

    public string ManagedExecutable => OperatingSystem.IsWindows()
        ? Path.Combine(ManagedRoot, "python", "python.exe")
        : Path.Combine(ManagedRoot, "python", "bin", "python3");

    public string WorkerDirectory => Path.Combine(Root, "worker");
    public string SelectionFile => Path.Combine(Root, "selection.json");
    public string DownloadDirectory => Path.Combine(Root, "downloads");

    public PythonPaths(string root) => Root = root;

    public static PythonPaths From(Configuration.OnnxStudioOptions options) => new(options.Python.ResolveDirectory());
}
