using System;
using System.Collections.Generic;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Mocks.Models;

namespace ONNXStudio.Mocks.ViewModels.Screens;

public partial class SettingsViewModel : ViewModelBase
{
    private readonly MainViewModel _mainViewModel;
    private AppSettings _settings;
    
    public SettingsViewModel(MainViewModel mainViewModel, AppSettings settings)
    {
        _mainViewModel = mainViewModel;
        _settings = settings;
        Title = "Settings";
        
        // Copy settings to allow editing
        Theme = settings.Theme;
        Language = settings.Language;
        StartupAction = settings.StartupAction;
        ApiPort = settings.ApiPort;
        MaxRequestSizeMb = settings.MaxRequestSizeMb;
        AllowAllOrigins = settings.AllowAllOrigins;
        LogLevel = settings.LogLevel;
        CacheSize = settings.CacheSize;
    }
    
    [ObservableProperty]
    private ThemeMode _theme;
    
    [ObservableProperty]
    private string _language;
    
    [ObservableProperty]
    private StartupAction _startupAction;
    
    [ObservableProperty]
    private int _apiPort;
    
    [ObservableProperty]
    private int _maxRequestSizeMb;
    
    [ObservableProperty]
    private bool _allowAllOrigins;
    
    [ObservableProperty]
    private LogLevel _logLevel;

    [ObservableProperty]
    private int _cacheSize;

    partial void OnThemeChanged(ThemeMode value)
    {
        // Apply immediately so the user sees the switch without pressing Save
        _mainViewModel.ApplyTheme(value);
    }
    
    [ObservableProperty]
    private string _statusMessage = string.Empty;
    
    public List<ThemeMode> Themes { get; } = new() { ThemeMode.Dark, ThemeMode.Light, ThemeMode.System };
    public List<string> Languages { get; } = new() { "English", "French", "German", "Spanish" };
    public List<StartupAction> StartupActions { get; } = new() { 
        StartupAction.OpenLastModel, 
        StartupAction.ShowWelcomeScreen, 
        StartupAction.ShowDashboard 
    };
    public List<LogLevel> LogLevels { get; } = new() { 
        LogLevel.Debug, LogLevel.Info, LogLevel.Warning, LogLevel.Error 
    };
    
    [RelayCommand]
    private void Back()
    {
        _mainViewModel.ShowDashboard();
    }
    
    [RelayCommand]
    private void Save()
    {
        // Update settings
        _settings.Theme = Theme;
        _settings.Language = Language;
        _settings.StartupAction = StartupAction;
        _settings.ApiPort = ApiPort;
        _settings.MaxRequestSizeMb = MaxRequestSizeMb;
        _settings.AllowAllOrigins = AllowAllOrigins;
        _settings.LogLevel = LogLevel;
        _settings.CacheSize = CacheSize;
        
        StatusMessage = "Settings saved successfully";
        _mainViewModel.ShowToast("Settings saved");
    }
    
    [RelayCommand]
    private void ClearCache()
    {
        StatusMessage = "Cache cleared";
    }
    
    public string Version => "1.0.0";
    
    [RelayCommand]
    private void ViewLicense()
    {
        StatusMessage = "Opening license...";
    }
    
    [RelayCommand]
    private void ViewSource()
    {
        StatusMessage = "Opening source repository...";
    }
}
