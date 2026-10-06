using System;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// Model loading screen (S-02): progress feedback while a model is parsed.
/// </summary>
public partial class ModelLoadingViewModel : ViewModelBase
{
    [ObservableProperty]
    private string _modelName = string.Empty;

    [ObservableProperty]
    private int _progressValue;

    [ObservableProperty]
    private string _statusMessage = "Opening file...";

    [ObservableProperty]
    private string? _errorMessage;

    /// <summary>Raised when the user cancels the load.</summary>
    public event Action? CancelRequested;

    public bool HasError => ErrorMessage != null;

    public ModelLoadingViewModel(string filePath)
    {
        Title = "Loading model";
        ModelName = System.IO.Path.GetFileName(filePath);
    }

    public void SetProgress(int percent, string status)
    {
        ProgressValue = percent;
        StatusMessage = status;
        OnPropertyChanged(nameof(HasError));
    }

    public void SetError(string message)
    {
        ErrorMessage = message;
        OnPropertyChanged(nameof(HasError));
    }

    [RelayCommand]
    private void Cancel()
    {
        CancelRequested?.Invoke();
    }
}
