using ONNXStudio.Core.Models;
using ONNXStudio.Core.Python;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Services;
using ONNXStudioUI.ViewModels.Screens;

namespace ONNXStudioUI.ViewModels;

/// <summary>
/// Orchestrates the model loading flow (S-02): loading screen with progress,
/// registry registration, toast feedback and navigation back to the dashboard.
/// Shared by the file picker, the drag-and-drop zone and the CLI argument.
/// </summary>
public interface IModelLoadCoordinator
{
    Task LoadAsync(string filePath);
    Task LoadManyAsync(IEnumerable<string> filePaths);

    /// <summary>Registers a model that was loaded by another importer (scikit-learn...): API start and toast, like an ONNX load.</summary>
    Task RegisterAsync(IModel model);
}

public sealed class ModelLoadCoordinator : IModelLoadCoordinator, IDisposable
{
    private readonly MainWindowViewModel _shell;
    private readonly IModelLoader _loader;
    private readonly IModelRegistry _registry;
    private readonly IToastService _toast;
    private readonly ONNXStudio.Api.ApiServerHost _apiHost;
    private readonly SemaphoreSlim _loadGate = new(1);
    private readonly object _cancelLock = new();
    private CancellationTokenSource? _activeCancellation;
    private int _disposed;

    public ModelLoadCoordinator(
        MainWindowViewModel shell,
        IModelLoader loader,
        IModelRegistry registry,
        IToastService toast,
        ONNXStudio.Api.ApiServerHost apiHost)
    {
        _shell = shell;
        _loader = loader;
        _registry = registry;
        _toast = toast;
        _apiHost = apiHost;
    }

    public Task LoadAsync(string filePath) => LoadManyAsync(new[] { filePath });

    public async Task RegisterAsync(IModel model)
    {
        _registry.Register(model);
        _toast.Show($"Model '{model.Name}' loaded");
        try { await _apiHost.StartAsync(); }
        catch (Exception)
        {
            _toast.Show("Model loaded. The API could not start; choose an available port in API settings.");
        }
    }

    private async Task<bool> LoadCoreAsync(string filePath)
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            return false;
        }

        var loading = new ModelLoadingViewModel(filePath);
        using var cancellation = new CancellationTokenSource();
        lock (_cancelLock)
        {
            if (_disposed != 0)
            {
                return false;
            }
            _activeCancellation = cancellation;
        }

        void Cancel()
        {
            cancellation.Cancel();
            if (_shell.Models.Count == 0) _shell.ShowWelcome();
            else _shell.ShowDashboard();
        }
        loading.CancelRequested += Cancel;

        _shell.SetCurrentScreen(loading);
        loading.SetProgress(10, "Validating file...");
        var hasError = false;

        try
        {
            var result = await Task.Run(() => _loader.LoadAsync(filePath, cancellation.Token));

            if (cancellation.IsCancellationRequested)
            {
                return false;
            }

            if (result.IsFailure)
            {
                var error = result.Error!;
                loading.SetError(error.Message);
                _toast.Show(error.Message);
                hasError = true;
                return false;
            }

            var model = result.Value!;
            loading.SetProgress(80, $"Parsed graph ({model.Graph.Nodes.Count} nodes)");
            _registry.Register(model);
            loading.SetProgress(100, "Done");

            _toast.Show($"Model '{model.Name}' loaded");
            try { await _apiHost.StartAsync(); }
            catch (Exception)
            {
                _toast.Show("Model loaded. The API could not start; choose an available port in API settings.");
            }
            _shell.ShowDashboard();
            return true;
        }
        catch (OperationCanceledException) when (cancellation.IsCancellationRequested) { }
        catch (Exception)
        {
            hasError = true;
            loading.SetError("The model could not be loaded. Check the file and try again.");
        }
        finally
        {
            lock (_cancelLock)
            {
                if (ReferenceEquals(_activeCancellation, cancellation))
                {
                    _activeCancellation = null;
                }
            }

            // Keep the error screen's Back/Cancel action usable after a failed load.
            loading.CancelRequested -= Cancel;
            if (hasError)
            {
                loading.CancelRequested += () => { if (_shell.Models.Count == 0) _shell.ShowWelcome(); else _shell.ShowDashboard(); };
            }
        }
        return false;
    }

    public async Task LoadManyAsync(IEnumerable<string> filePaths)
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            return;
        }

        var paths = filePaths.ToList();

        // joblib / pickle files open their own screen instead of the ONNX loading flow.
        foreach (var path in paths.Where(PythonModel.IsPythonModelFile)) _shell.OpenPythonModel(path);

        try
        {
            await _loadGate.WaitAsync();
        }
        catch (ObjectDisposedException)
        {
            return;
        }

        try
        {
            foreach (var path in paths.Where(p => !PythonModel.IsPythonModelFile(p)))
            {
                if (Volatile.Read(ref _disposed) != 0 || !await LoadCoreAsync(path)) break;
            }
        }
        finally
        {
            try
            {
                _loadGate.Release();
            }
            catch (ObjectDisposedException)
            {
            }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        CancellationTokenSource? active;
        lock (_cancelLock)
        {
            active = _activeCancellation;
            _activeCancellation = null;
        }

        active?.Cancel();
        _loadGate.Dispose();
    }
}
