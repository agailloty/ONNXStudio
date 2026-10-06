using Microsoft.ML.OnnxRuntime;

namespace ONNXStudio.Core.Services;

/// <summary>
/// Owns the ONNX Runtime inference sessions, one per loaded model, with a
/// least-recently-used cache (capacity from OnnxStudioOptions.SessionCacheSize).
/// </summary>
public interface IInferenceSessionManager : IDisposable
{
    /// <summary>Gets (and caches) the session of a model. Thread-safe.</summary>
    InferenceSession GetSession(Models.OnnxModel model);

    /// <summary>Evicts the session of a model (called when a model is unloaded).</summary>
    void Evict(string filePath);

    int CachedSessionCount { get; }
}

public sealed class InferenceSessionManager : IInferenceSessionManager
{
    private readonly object _sync = new();
    private readonly int _capacity;
    private readonly Dictionary<string, LinkedListNode<(string Path, InferenceSession Session)>> _cache = new();
    private readonly LinkedList<(string Path, InferenceSession Session)> _lru = new();

    public InferenceSessionManager(Microsoft.Extensions.Options.IOptions<Configuration.OnnxStudioOptions> options)
    {
        _capacity = Math.Max(1, options.Value.SessionCacheSize);
    }

    public int CachedSessionCount
    {
        get { lock (_sync) { return _cache.Count; } }
    }

    public InferenceSession GetSession(Models.OnnxModel model)
    {
        lock (_sync)
        {
            if (_cache.TryGetValue(model.FilePath, out var node))
            {
                _lru.Remove(node);
                _lru.AddFirst(node);
                return node.Value.Session;
            }

            var session = new InferenceSession(model.FilePath);
            var newNode = _lru.AddFirst((model.FilePath, session));
            _cache[model.FilePath] = newNode;

            while (_lru.Count > _capacity)
            {
                var last = _lru.Last!;
                _lru.RemoveLast();
                _cache.Remove(last.Value.Path);
                last.Value.Session.Dispose();
            }

            return session;
        }
    }

    public void Evict(string filePath)
    {
        lock (_sync)
        {
            if (_cache.Remove(filePath, out var node))
            {
                _lru.Remove(node);
                node.Value.Session.Dispose();
            }
        }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            foreach (var node in _cache.Values)
            {
                node.Value.Session.Dispose();
            }
            _cache.Clear();
            _lru.Clear();
        }
    }
}
