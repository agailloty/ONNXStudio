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

public sealed class ToastService : IToastService
{
    private CancellationTokenSource? _cts;

    public string? Current { get; private set; }
    public event Action? ToastChanged;

    public void Show(string message)
    {
        Current = message;
        ToastChanged?.Invoke();

        _cts?.Cancel();
        _cts = new CancellationTokenSource();
        _ = AutoHideAsync(_cts.Token);
    }

    private async Task AutoHideAsync(CancellationToken ct)
    {
        try
        {
            await Task.Delay(3000, ct).ConfigureAwait(true);
            Current = null;
            ToastChanged?.Invoke();
        }
        catch (TaskCanceledException)
        {
            // replaced by a newer toast
        }
    }
}
