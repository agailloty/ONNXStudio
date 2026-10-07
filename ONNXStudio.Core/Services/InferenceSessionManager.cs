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

    /// <summary>Pins a session until the inference operation has completed.</summary>
    InferenceSessionLease Acquire(Models.OnnxModel model);

    /// <summary>Evicts the session of a model (called when a model is unloaded).</summary>
    void Evict(string filePath);

    int CachedSessionCount { get; }
}

public sealed class InferenceSessionLease : IDisposable
{
    private Action? _release;
    public InferenceSession Session { get; }
    internal InferenceSessionLease(InferenceSession session, Action release) { Session = session; _release = release; }
    public void Dispose() => Interlocked.Exchange(ref _release, null)?.Invoke();
}

public sealed class InferenceSessionManager : IInferenceSessionManager
{
    private readonly object _sync = new();
    private readonly int _capacity;
    private readonly Dictionary<string, LinkedListNode<(string Path, InferenceSession Session)>> _cache = new();
    private readonly LinkedList<(string Path, InferenceSession Session)> _lru = new();
    private readonly Dictionary<InferenceSession, int> _leases = new();
    private readonly HashSet<InferenceSession> _retired = new();
    private readonly IModelRegistry? _registry;
    private bool _disposed;

    public InferenceSessionManager(Microsoft.Extensions.Options.IOptions<Configuration.OnnxStudioOptions> options, IModelRegistry? registry = null)
    {
        _capacity = Math.Max(1, options.Value.SessionCacheSize);
        _registry = registry;
        if (_registry != null) _registry.ModelRemoved += OnModelRemoved;
    }

    public int CachedSessionCount
    {
        get { lock (_sync) { return _cache.Count; } }
    }

    public InferenceSession GetSession(Models.OnnxModel model)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
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
                Retire(last.Value.Session);
            }

            return session;
        }
    }

    public InferenceSessionLease Acquire(Models.OnnxModel model)
    {
        lock (_sync)
        {
            var session = GetSession(model);
            _leases[session] = _leases.GetValueOrDefault(session) + 1;
            return new InferenceSessionLease(session, () => Release(session));
        }
    }

    private void Release(InferenceSession session)
    {
        lock (_sync)
        {
            if (--_leases[session] != 0) return;
            _leases.Remove(session);
            if (_retired.Remove(session)) session.Dispose();
        }
    }

    private void Retire(InferenceSession session)
    {
        if (_leases.ContainsKey(session)) _retired.Add(session);
        else session.Dispose();
    }

    private void OnModelRemoved(object? sender, Models.IModel model) => Evict(model.FilePath);

    public void Evict(string filePath)
    {
        lock (_sync)
        {
            if (_cache.Remove(filePath, out var node))
            {
                _lru.Remove(node);
                Retire(node.Value.Session);
            }
        }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
            if (_registry != null) _registry.ModelRemoved -= OnModelRemoved;
            foreach (var node in _cache.Values)
            {
                Retire(node.Value.Session);
            }
            _cache.Clear();
            _lru.Clear();
        }
    }
}
