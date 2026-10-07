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
}

public sealed class ModelLoadCoordinator : IModelLoadCoordinator
{
    private readonly MainWindowViewModel _shell;
    private readonly IModelLoader _loader;
    private readonly IModelRegistry _registry;
    private readonly IToastService _toast;
    private readonly ONNXStudio.Api.ApiServerHost _apiHost;
    private readonly SemaphoreSlim _loadGate = new(1);

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

    private async Task<bool> LoadCoreAsync(string filePath)
    {
        var loading = new ModelLoadingViewModel(filePath);
        using var cancellation = new CancellationTokenSource();
        void Cancel()
        {
            cancellation.Cancel();
            if (_shell.Models.Count == 0) _shell.ShowWelcome();
            else _shell.ShowDashboard();
        }
        loading.CancelRequested += Cancel;

        _shell.SetCurrentScreen(loading);
        loading.SetProgress(10, "Validating file...");

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
            loading.SetError("The model could not be loaded. Check the file and try again.");
        }
        finally
        {
            // Keep the error screen's Back/Cancel action usable after loading finishes.
            loading.CancelRequested -= Cancel;
            loading.CancelRequested += () => { if (_shell.Models.Count == 0) _shell.ShowWelcome(); else _shell.ShowDashboard(); };
        }
        return false;
    }

    public async Task LoadManyAsync(IEnumerable<string> filePaths)
    {
        var paths = filePaths.ToList();

        // joblib / pickle files open their own screen instead of the ONNX loading flow.
        foreach (var path in paths.Where(PythonModel.IsPythonModelFile)) _shell.OpenPythonModel(path);

        await _loadGate.WaitAsync();
        try
        {
            foreach (var path in paths.Where(p => !PythonModel.IsPythonModelFile(p)))
            {
                if (!await LoadCoreAsync(path)) break;
            }
        }
        finally { _loadGate.Release(); }
    }
}
