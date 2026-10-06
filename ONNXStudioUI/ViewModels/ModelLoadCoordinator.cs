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

    public ModelLoadCoordinator(
        MainWindowViewModel shell,
        IModelLoader loader,
        IModelRegistry registry,
        IToastService toast)
    {
        _shell = shell;
        _loader = loader;
        _registry = registry;
        _toast = toast;
    }

    public async Task LoadAsync(string filePath)
    {
        var loading = new ModelLoadingViewModel(filePath);
        var cancelled = false;
        loading.CancelRequested += () => { cancelled = true; };

        _shell.SetCurrentScreen(loading);
        loading.SetProgress(10, "Validating file...");

        var result = await _loader.LoadAsync(filePath).ConfigureAwait(true);

        if (cancelled)
        {
            _shell.ShowWelcome();
            return;
        }

        if (result.IsFailure)
        {
            var error = result.Error!;
            loading.SetError(error.Message);
            _toast.Show(error.Message);
            return;
        }

        var model = result.Value!;
        loading.SetProgress(80, $"Parsed graph ({model.Graph.Nodes.Count} nodes)");
        _registry.Register(model);
        loading.SetProgress(100, "Done");

        _toast.Show($"Model '{model.Name}' loaded");
        _shell.ShowDashboard();
    }

    public async Task LoadManyAsync(IEnumerable<string> filePaths)
    {
        foreach (var path in filePaths)
        {
            await LoadAsync(path).ConfigureAwait(true);
        }
    }
}
