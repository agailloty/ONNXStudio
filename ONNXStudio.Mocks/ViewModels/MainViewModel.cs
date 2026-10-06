using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Styling;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Mocks.Models;
using ONNXStudio.Mocks.ViewModels.Screens;
using ThemeMode = ONNXStudio.Mocks.Models.ThemeMode;

namespace ONNXStudio.Mocks.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    private readonly Dictionary<string, ViewModelBase> _screenCache = new();
    private SettingsViewModel? _settingsScreen;
    private CancellationTokenSource? _toastCts;

    [ObservableProperty]
    private ViewModelBase? _currentViewModel;

    [ObservableProperty]
    private ObservableCollection<OnnxModel> _models = new();

    [ObservableProperty]
    private OnnxModel? _selectedModel;

    [ObservableProperty]
    private AppSettings _settings = new();

    [ObservableProperty]
    private string _statusMessage = "Ready";

    [ObservableProperty]
    private string? _toast;

    public MainViewModel()
    {
        Title = "ONNX Studio";
        LoadSettings();
        ApplyTheme(Settings.Theme);

        // Initialize with mock models for demo
        if (Models.Count == 0)
        {
            // Add some mock models
            Models.Add(MockDataGenerator.CreateMockResNet50());
            Models.Add(MockDataGenerator.CreateMockBert());
            Models.Add(MockDataGenerator.CreateMockMobileNet());
            Models.Add(MockDataGenerator.CreateMockCaliforniaHousingPipeline());
        }

        // Start with dashboard if models exist, otherwise welcome
        if (Models.Count > 0)
        {
            CurrentViewModel = new DashboardViewModel(this);
        }
        else
        {
            CurrentViewModel = new WelcomeViewModel(this);
        }
    }

    public void NavigateTo(ViewModelBase viewModel)
    {
        CurrentViewModel = viewModel;
    }

    [RelayCommand]
    public void ShowWelcome()
    {
        CurrentViewModel = new WelcomeViewModel(this);
    }

    [RelayCommand]
    public void ShowDashboard()
    {
        CurrentViewModel = new DashboardViewModel(this);
    }

    [RelayCommand]
    public void ShowModelLoading(string filePath)
    {
        CurrentViewModel = new ModelLoadingViewModel(this, filePath);
    }

    [RelayCommand]
    public void ShowModelInspector(OnnxModel model)
    {
        SelectedModel = model;
        CurrentViewModel = GetOrCreateScreen(model, () => new ModelInspectorViewModel(this, model));
    }

    [RelayCommand]
    public void ShowInferencePlayground(OnnxModel model)
    {
        SelectedModel = model;
        CurrentViewModel = GetOrCreateScreen(model, () => new InferencePlaygroundViewModel(this, model));
    }

    [RelayCommand]
    public void ShowApiSandbox(OnnxModel model)
    {
        SelectedModel = model;
        CurrentViewModel = GetOrCreateScreen(model, () => new ApiSandboxViewModel(this, model));
    }

    [RelayCommand]
    public void ShowApiConfig(OnnxModel model)
    {
        SelectedModel = model;
        CurrentViewModel = GetOrCreateScreen(model, () => new ApiConfigViewModel(this, model));
    }

    [RelayCommand]
    public void ShowSettings()
    {
        _settingsScreen ??= new SettingsViewModel(this, Settings);
        CurrentViewModel = _settingsScreen;
    }

    /// <summary>
    /// Caches model screens so switching between Inspect / Inference / API
    /// tabs preserves the state of each screen (results, selections, mappings...).
    /// </summary>
    private T GetOrCreateScreen<T>(OnnxModel model, Func<T> factory) where T : ViewModelBase
    {
        var key = typeof(T).Name + ":" + model.Id;
        if (_screenCache.TryGetValue(key, out var cached))
        {
            return (T)cached;
        }

        var vm = factory();
        _screenCache[key] = vm;
        return vm;
    }

    /// <summary>
    /// Shows a transient toast message (auto-hides after 3 seconds).
    /// </summary>
    public void ShowToast(string message)
    {
        Toast = message;
        _toastCts?.Cancel();
        _toastCts = new CancellationTokenSource();
        _ = AutoHideToastAsync(_toastCts.Token);
    }

    private async Task AutoHideToastAsync(CancellationToken ct)
    {
        try
        {
            await Task.Delay(3000, ct);
            if (!ct.IsCancellationRequested)
            {
                Toast = null;
            }
        }
        catch (TaskCanceledException)
        {
            // A newer toast replaced this one
        }
    }

    public void AddModel(OnnxModel model)
    {
        Models.Add(model);
        SelectedModel = model;
        StatusMessage = $"Model '{model.Name}' loaded successfully";
        ShowToast($"Model '{model.Name}' loaded");
    }

    public void RemoveModel(OnnxModel model)
    {
        Models.Remove(model);
        if (SelectedModel == model)
        {
            SelectedModel = Models.FirstOrDefault();
        }
        StatusMessage = $"Model '{model.Name}' unloaded";
        ShowToast($"Model '{model.Name}' unloaded");

        // Evict cached screens of the removed model
        var keys = _screenCache.Keys.Where(k => k.EndsWith(":" + model.Id)).ToList();
        foreach (var key in keys)
        {
            _screenCache.Remove(key);
        }
    }

    [RelayCommand]
    public void BackToDashboard()
    {
        if (Models.Count > 0)
        {
            ShowDashboard();
        }
        else
        {
            ShowWelcome();
        }
    }

    private void LoadSettings()
    {
        // TODO: Load from config file
        // For now, use defaults
        Settings = new AppSettings();
    }

    /// <summary>
    /// Applies the given theme to the whole application (FluentTheme + custom palettes).
    /// </summary>
    public void ApplyTheme(ThemeMode theme)
    {
        Settings.Theme = theme;

        if (Application.Current is { } app)
        {
            app.RequestedThemeVariant = theme switch
            {
                ThemeMode.Dark => ThemeVariant.Dark,
                ThemeMode.Light => ThemeVariant.Light,
                _ => null // System
            };
        }
    }

    public void SaveSettings()
    {
        // TODO: Save to config file
        StatusMessage = "Settings saved";
        ShowToast("Settings saved");
    }
}
