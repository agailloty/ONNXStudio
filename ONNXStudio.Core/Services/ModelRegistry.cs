using System.Collections.Concurrent;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Services;

/// <summary>
/// Central repository of loaded models (US-001: multiple models simultaneously,
/// each with a unique ModelId).
/// </summary>
public interface IModelRegistry
{
    event EventHandler<IModel>? ModelAdded;
    event EventHandler<IModel>? ModelRemoved;

    IReadOnlyList<IModel> Models { get; }
    IModel? GetById(string id);
    bool IsLoaded(string filePath);

    /// <summary>Registers a loaded model. Reloads of the same file replace the previous entry.</summary>
    IModel Register(IModel model);

    /// <summary>Removes a model. Returns true if the model was registered.</summary>
    bool Unload(string modelId);
}

public sealed class ModelRegistry : IModelRegistry
{
    private readonly ConcurrentDictionary<string, IModel> _models = new();
    private readonly object _sync = new();

    public event EventHandler<IModel>? ModelAdded;
    public event EventHandler<IModel>? ModelRemoved;

    public IReadOnlyList<IModel> Models
    {
        get
        {
            lock (_sync)
            {
                return _models.Values.OrderBy(m => m.LoadedAt).ToList();
            }
        }
    }

    public IModel? GetById(string id)
        => _models.TryGetValue(id, out var model) ? model : null;

    public bool IsLoaded(string filePath)
    {
        lock (_sync)
        {
            return _models.Values.Any(m =>
                string.Equals(m.FilePath, filePath, StringComparison.OrdinalIgnoreCase));
        }
    }

    public IModel Register(IModel model)
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
        IModel? removed = null;
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
