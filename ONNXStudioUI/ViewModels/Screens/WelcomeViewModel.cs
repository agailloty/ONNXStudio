using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// Empty-state welcome screen (S-01): open a model via the file picker or
/// drag-and-drop. Loading itself is handled by the ModelLoadCoordinator.
/// </summary>
public partial class WelcomeViewModel : ViewModelBase
{
    private readonly IModelLoadCoordinator _loadCoordinator;
    private readonly IFilePickerService _filePicker;

    [ObservableProperty]
    private string? _statusMessage;

    public WelcomeViewModel(
        IModelLoadCoordinator loadCoordinator,
        IFilePickerService filePicker)
    {
        _loadCoordinator = loadCoordinator;
        _filePicker = filePicker;
        Title = "Welcome";
        StatusMessage = "Drop an .onnx, .joblib or .pkl file or click Open Model";
    }

    [RelayCommand]
    private async Task OpenModelAsync()
    {
        IsBusy = true;
        StatusMessage = "Opening file picker...";
        try
        {
            var paths = await _filePicker.PickModelFilesAsync().ConfigureAwait(true);
            if (paths.Length == 0)
            {
                StatusMessage = "No model selected";
                return;
            }
            await _loadCoordinator.LoadManyAsync(paths).ConfigureAwait(true);
        }
        finally
        {
            IsBusy = false;
        }
    }

    /// <summary>Called by the drag-and-drop handler of the view.</summary>
    [RelayCommand]
    private async Task DropModelAsync(string filePath)
    {
        IsBusy = true;
        try
        {
            await _loadCoordinator.LoadAsync(filePath).ConfigureAwait(true);
        }
        finally
        {
            IsBusy = false;
        }
    }
}
