using System;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Mocks.Models;

namespace ONNXStudio.Mocks.ViewModels.Screens;

public partial class ModelLoadingViewModel : ViewModelBase
{
    private readonly MainViewModel _mainViewModel;
    private readonly string _filePath;
    private bool _cancelled;

    [ObservableProperty]
    private string _modelName = "Loading model...";

    [ObservableProperty]
    private int _progressValue;

    [ObservableProperty]
    private string _progressText = "0%";

    [ObservableProperty]
    private string _statusMessage = "Parsing graph...";

    [ObservableProperty]
    private bool _showError = false;

    [ObservableProperty]
    private string _errorMessage = string.Empty;

    public ModelLoadingViewModel(MainViewModel mainViewModel, string filePath)
    {
        _mainViewModel = mainViewModel;
        _filePath = filePath;
        Title = "Loading Model";

        ModelName = System.IO.Path.GetFileName(filePath);

        _ = LoadModelAsync();
    }

    private async Task LoadModelAsync()
    {
        try
        {
            var steps = new (int percent, string status)[]
            {
                (10, "Opening file..."),
                (30, "Parsing graph... (175 nodes)"),
                (60, "Loading weights... (23 MB)"),
                (80, "Validating model..."),
                (100, "Finalizing...")
            };

            foreach (var (percent, status) in steps)
            {
                await Task.Delay(250);
                if (_cancelled) return;

                ProgressValue = percent;
                ProgressText = percent + "%";
                StatusMessage = status;
            }

            await Task.Delay(400);
            if (_cancelled) return;

            // Model loaded successfully - create mock model and navigate to inspector
            var model = System.IO.Path.HasExtension(_filePath)
                ? MockDataGenerator.CreateMockSimpleModel()
                : MockDataGenerator.CreateMockResNet50();
            model.FilePath = _filePath;

            _mainViewModel.AddModel(model);
            _mainViewModel.ShowModelInspector(model);
        }
        catch (Exception ex)
        {
            ShowError = true;
            ErrorMessage = "Failed to load model: " + ex.Message;
            StatusMessage = "Error";
        }
    }

    [RelayCommand]
    private void Cancel()
    {
        _cancelled = true;
        _mainViewModel.BackToDashboard();
    }

    [RelayCommand]
    private void Retry()
    {
        ShowError = false;
        ErrorMessage = string.Empty;
        ProgressValue = 0;
        ProgressText = "0%";
        StatusMessage = "Retrying...";
        _ = LoadModelAsync();
    }
}
