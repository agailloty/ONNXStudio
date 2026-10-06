using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// Empty-state welcome screen (S-01): encourages loading a first model.
/// </summary>
public partial class WelcomeViewModel : ViewModelBase
{
    private readonly MainWindowViewModel _shell;
    private readonly IModelLoader _loader;
    private readonly IModelRegistry _registry;
    private readonly IToastService _toast;
    private readonly IFilePickerService _filePicker;

    [ObservableProperty]
    private string? _statusMessage;

    public WelcomeViewModel(
        MainWindowViewModel shell,
        IModelLoader loader,
        IModelRegistry registry,
        IToastService toast,
        IFilePickerService filePicker)
    {
        _shell = shell;
        _loader = loader;
        _registry = registry;
        _toast = toast;
        _filePicker = filePicker;
        Title = "Welcome";
        StatusMessage = "Drop an .onnx file or click Open Model";
    }

    [RelayCommand]
    private async Task OpenModelAsync()
    {
        IsBusy = true;
        StatusMessage = "Opening file picker...";
        try
        {
            var path = await _filePicker.PickModelFileAsync().ConfigureAwait(true);
            if (path == null)
            {
                StatusMessage = "No model selected";
                return;
            }

            StatusMessage = $"Loading '{path}'...";
            await LoadModelAsync(path).ConfigureAwait(true);
        }
        finally
        {
            IsBusy = false;
        }
    }

    public async Task LoadModelAsync(string path)
    {
        var result = await _loader.LoadAsync(path).ConfigureAwait(true);
        if (result.IsSuccess)
        {
            _registry.Register(result.Value!);
            _toast.Show($"Model '{result.Value!.Name}' loaded");
            StatusMessage = $"Model '{result.Value.Name}' loaded";
            _shell.ShowDashboard();
        }
        else
        {
            var error = result.Error!;
            _toast.Show(error.Message);
            StatusMessage = error.Message;
        }
    }
}
