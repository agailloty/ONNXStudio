using System.Collections.Concurrent;

namespace ONNXStudio.Core.Python;

/// <summary>Holds the joblib / pickle models opened in the studio.</summary>
public interface IPythonModelRegistry
{
    event EventHandler<PythonModel>? ModelAdded;
    event EventHandler<PythonModel>? ModelRemoved;

    IReadOnlyList<PythonModel> Models { get; }

    /// <summary>Registers the file; opening the same file twice returns the existing entry.</summary>
    PythonModel Register(string filePath);

    bool Unload(string id);
}

public sealed class PythonModelRegistry : IPythonModelRegistry
{
    private readonly ConcurrentDictionary<string, PythonModel> _models = new();

    public event EventHandler<PythonModel>? ModelAdded;
    public event EventHandler<PythonModel>? ModelRemoved;

    public IReadOnlyList<PythonModel> Models => _models.Values.OrderBy(m => m.LoadedAt).ToArray();

    public PythonModel Register(string filePath)
    {
        var fullPath = Path.GetFullPath(filePath);
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        var existing = _models.Values.FirstOrDefault(m => string.Equals(m.FilePath, fullPath, comparison));
        if (existing != null) return existing;

        var size = File.Exists(fullPath) ? new FileInfo(fullPath).Length : 0;
        var model = new PythonModel(Guid.NewGuid().ToString("N"), fullPath, size);
        _models[model.Id] = model;
        ModelAdded?.Invoke(this, model);
        return model;
    }

    public bool Unload(string id)
    {
        if (!_models.TryRemove(id, out var model)) return false;
        ModelRemoved?.Invoke(this, model);
        return true;
    }
}
