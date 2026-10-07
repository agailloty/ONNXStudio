using System;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.DependencyInjection;
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
    private readonly IGraphAnalysisService _graphService;
    private readonly IToastService _toast;
    private readonly IThemeService _theme;
    private readonly ILogger<MainWindowViewModel> _logger;
    private readonly Dictionary<string, ViewModelBase> _screenCache = new();

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
        IGraphAnalysisService graphService,
        IToastService toast,
        IThemeService theme,
        ILogger<MainWindowViewModel> logger)
    {
        _services = services;
        _registry = registry;
        _graphService = graphService;
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

    public void ShowSettings()
    {
        CurrentViewModel = new ViewModels.Screens.SettingsViewModel(
            this,
            _services.GetRequiredService<IThemeService>(),
            _services.GetRequiredService<IToastService>(),
            _services.GetRequiredService<ONNXStudio.Api.ApiServerHost>(),
            _services.GetRequiredService<SettingsStore>());
    }

    /// <summary>
    /// Sets the current screen directly (used by the model load coordinator).
    /// </summary>
    public void SetCurrentScreen(ViewModelBase viewModel)
    {
        CurrentViewModel = viewModel;
    }

    // ----- model screens (cached per model so state is preserved) -----

    public void ShowInspector(OnnxModel model)
    {
        CurrentViewModel = GetOrCreateScreen("inspector:" + model.Id,
            () => new ViewModels.Screens.ModelInspectorViewModel(this, _graphService, model));
    }

    // Playground navigation is wired in T9; API screens in T10.
    public void ShowPlayground(OnnxModel model)
    {
        CurrentViewModel = GetOrCreateScreen("playground:" + model.Id,
            () => new ViewModels.Screens.InferencePlaygroundViewModel(
                this,
                _services.GetRequiredService<IInferenceService>(),
                _services.GetRequiredService<IFormGenerationService>(),
                _services.GetRequiredService<IFilePickerService>(),
                model));
    }

    public void ShowApiConfig(OnnxModel model)
    {
        CurrentViewModel = GetOrCreateScreen("apiconfig:" + model.Id,
            () => new ViewModels.Screens.ApiConfigViewModel(
                this,
                _services.GetRequiredService<ONNXStudio.Api.ApiServerHost>(),
                _services.GetRequiredService<IToastService>(),
                model));
    }

    public void ShowApiSandbox(OnnxModel model)
    {
        CurrentViewModel = GetOrCreateScreen("sandbox:" + model.Id,
            () => new ViewModels.Screens.ApiSandboxViewModel(
                this,
                _services.GetRequiredService<ONNXStudio.Api.ApiServerHost>(),
                _services.GetRequiredService<IToastService>(),
                model));
    }

    private ViewModelBase GetOrCreateScreen(string key, Func<ViewModelBase> factory)
    {
        if (_screenCache.TryGetValue(key, out var cached))
        {
            return cached;
        }
        var vm = factory();
        _screenCache[key] = vm;
        return vm;
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
            foreach (var key in _screenCache.Keys.Where(k => k.EndsWith(":" + model.Id, StringComparison.Ordinal)).ToArray())
            {
                if (_screenCache.Remove(key, out var screen))
                {
                    if (screen is IDisposable disposable) disposable.Dispose();
                    if (ReferenceEquals(CurrentViewModel, screen)) ShowDashboard();
                }
            }
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
        Avalonia.Threading.Dispatcher.UIThread.Post(() => ToastMessage = _toast.Current);
    }

    public void ShowToast(string message) => _toast.Show(message);

    public void Dispose()
    {
        _registry.ModelAdded -= OnModelAdded;
        _registry.ModelRemoved -= OnModelRemoved;
        _toast.ToastChanged -= OnToastChanged;
        foreach (var screen in _screenCache.Values.OfType<IDisposable>()) screen.Dispose();
        _screenCache.Clear();
    }
}
