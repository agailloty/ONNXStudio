using System.Collections.Concurrent;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Services;

/// <summary>
/// Central repository of loaded models (US-001: multiple models simultaneously,
/// each with a unique ModelId).
/// </summary>
public interface IModelRegistry
{
    event EventHandler<OnnxModel>? ModelAdded;
    event EventHandler<OnnxModel>? ModelRemoved;

    IReadOnlyList<OnnxModel> Models { get; }
    OnnxModel? GetById(string id);
    bool IsLoaded(string filePath);

    /// <summary>Registers a loaded model. Reloads of the same file replace the previous entry.</summary>
    OnnxModel Register(OnnxModel model);

    /// <summary>Removes a model. Returns true if the model was registered.</summary>
    bool Unload(string modelId);
}

public sealed class ModelRegistry : IModelRegistry
{
    private readonly ConcurrentDictionary<string, OnnxModel> _models = new();
    private readonly object _sync = new();

    public event EventHandler<OnnxModel>? ModelAdded;
    public event EventHandler<OnnxModel>? ModelRemoved;

    public IReadOnlyList<OnnxModel> Models
    {
        get
        {
            lock (_sync)
            {
                return _models.Values.OrderBy(m => m.LoadedAt).ToList();
            }
        }
    }

    public OnnxModel? GetById(string id)
        => _models.TryGetValue(id, out var model) ? model : null;

    public bool IsLoaded(string filePath)
    {
        lock (_sync)
        {
            return _models.Values.Any(m =>
                string.Equals(m.FilePath, filePath, StringComparison.OrdinalIgnoreCase));
        }
    }

    public OnnxModel Register(OnnxModel model)
    {
        lock (_sync)
        {
            // Replace an existing registration of the same file
            var existing = _models.Values.FirstOrDefault(m =>
                string.Equals(m.FilePath, model.FilePath, StringComparison.OrdinalIgnoreCase));
            if (existing != null)
            {
                _models.TryRemove(existing.Id, out _);
                ModelRemoved?.Invoke(this, existing);
            }

            _models[model.Id] = model;
        }

        ModelAdded?.Invoke(this, model);
        return model;
    }

    public bool Unload(string modelId)
    {
        OnnxModel? removed = null;
        lock (_sync)
        {
            if (_models.TryRemove(modelId, out var model))
            {
                removed = model;
            }
        }

        if (removed != null)
        {
            ModelRemoved?.Invoke(this, removed);
            return true;
        }
        return false;
    }
}
