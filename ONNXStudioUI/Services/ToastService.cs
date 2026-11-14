using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;

namespace ONNXStudioUI.Services;

/// <summary>
/// Transient toast messages shown at the top of the main window (3 s auto-hide).
/// </summary>
public interface IToastService
{
    string? Current { get; }
    event Action? ToastChanged;
    void Show(string message);
}

public sealed class ToastService : IToastService, IDisposable
{
    private readonly object _sync = new();
    private CancellationTokenSource? _cts;
    private bool _disposed;

    public string? Current { get; private set; }
    public event Action? ToastChanged;

    public void Show(string message)
    {
        CancellationTokenSource next;
        CancellationToken token;
        CancellationTokenSource? previous;
        lock (_sync)
        {
            if (_disposed) return;
            previous = _cts;
            next = new CancellationTokenSource();
            token = next.Token;
            _cts = next;
        }

        previous?.Cancel();
        previous?.Dispose();

        Current = message;
        ToastChanged?.Invoke();
        _ = AutoHideAsync(next, token);
    }

    private async Task AutoHideAsync(CancellationTokenSource owner, CancellationToken ct)
    {
        try
        {
            await Task.Delay(3000, ct).ConfigureAwait(true);
            lock (_sync)
            {
                if (_disposed || !ReferenceEquals(_cts, owner)) return;
            }
            Current = null;
            ToastChanged?.Invoke();
        }
        catch (OperationCanceledException)
        {
            // replaced by a newer toast or disposed
        }
    }

    public void Dispose()
    {
        CancellationTokenSource? cts;
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
            cts = _cts;
            _cts = null;
        }

        cts?.Cancel();
        cts?.Dispose();
    }
}
