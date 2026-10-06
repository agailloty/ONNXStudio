using System;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.ViewModels;

/// <summary>
/// Shell view model: navigation between screens, model registry projection,
/// status bar, toasts and theme. Screen view models are created through DI.
/// </summary>
public partial class MainWindowViewModel : ObservableObject, IDisposable
{
    private readonly IServiceProvider _services;
    private readonly IModelRegistry _registry;
    private readonly IModelLoader _loader;
    private readonly IToastService _toast;
    private readonly IThemeService _theme;
    private readonly ILogger<MainWindowViewModel> _logger;

    [ObservableProperty]
    private ViewModelBase? _currentViewModel;

    [ObservableProperty]
    private ObservableCollection<OnnxModel> _models = new();

    [ObservableProperty]
    private string _statusMessage = "Ready";

    [ObservableProperty]
    private string? _toastMessage;

    public MainWindowViewModel(
        IServiceProvider services,
        IModelRegistry registry,
        IModelLoader loader,
        IToastService toast,
        IThemeService theme,
        ILogger<MainWindowViewModel> logger)
    {
        _services = services;
        _registry = registry;
        _loader = loader;
        _toast = toast;
        _theme = theme;
        _logger = logger;

        _registry.ModelAdded += OnModelAdded;
        _registry.ModelRemoved += OnModelRemoved;
        _toast.ToastChanged += OnToastChanged;

        RefreshModels();
        // NOTE: navigation is triggered by the App after this singleton is fully
        // constructed (screen view models depend on this shell view model).
    }

    public string AppVersion => "1.0.0";

    // ----- navigation -----

    [RelayCommand]
    public void ShowWelcome()
    {
        CurrentViewModel = _services.GetService(typeof(ViewModels.Screens.WelcomeViewModel)) is ViewModels.Screens.WelcomeViewModel welcome
            ? welcome
            : null;
        StatusMessage = Models.Count == 0 ? "No model loaded" : $"{Models.Count} model(s) loaded";
    }

    [RelayCommand]
    public void ShowDashboard()
    {
        CurrentViewModel = _services.GetService(typeof(ViewModels.Screens.DashboardViewModel)) is ViewModels.Screens.DashboardViewModel dashboard
            ? dashboard
            : null;
    }

    /// <summary>
    /// Sets the current screen directly (used by the model load coordinator).
    /// </summary>
    public void SetCurrentScreen(ViewModelBase viewModel)
    {
        CurrentViewModel = viewModel;
    }

    // ----- model registry projection -----

    private void OnModelAdded(object? sender, OnnxModel model)
    {
        Avalonia.Threading.Dispatcher.UIThread.Post(() =>
        {
            Models.Add(model);
            StatusMessage = $"Model '{model.Name}' loaded";
        });
    }

    private void OnModelRemoved(object? sender, OnnxModel model)
    {
        Avalonia.Threading.Dispatcher.UIThread.Post(() =>
        {
            Models.Remove(model);
            StatusMessage = $"Model '{model.Name}' unloaded";
        });
    }

    private void RefreshModels()
    {
        Models = new ObservableCollection<OnnxModel>(_registry.Models);
    }

    // ----- toasts -----

    private void OnToastChanged()
    {
        ToastMessage = _toast.Current;
    }

    public void ShowToast(string message) => _toast.Show(message);

    public void Dispose()
    {
        _registry.ModelAdded -= OnModelAdded;
        _registry.ModelRemoved -= OnModelRemoved;
        _toast.ToastChanged -= OnToastChanged;
    }
}
