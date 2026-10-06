using System;
using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Api;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// Application settings (S-07): theme, API port and about section.
/// </summary>
public partial class SettingsViewModel : ViewModelBase
{
    private readonly MainWindowViewModel _shell;
    private readonly IThemeService _theme;
    private readonly IToastService _toast;
    private readonly ApiServerHost _apiHost;

    [ObservableProperty]
    private AppTheme _selectedTheme;

    [ObservableProperty]
    private int _apiPort;

    public IReadOnlyList<AppTheme> Themes { get; } = new[] { AppTheme.Light, AppTheme.Dark, AppTheme.System };
    public string AppVersion => "1.0.0";

    public SettingsViewModel(MainWindowViewModel shell, IThemeService theme, IToastService toast, ApiServerHost apiHost)
    {
        _shell = shell;
        _theme = theme;
        _toast = toast;
        _apiHost = apiHost;
        Title = "Settings";

        SelectedTheme = _theme.Current;
        ApiPort = apiHost.RequestedPort;
    }

    partial void OnSelectedThemeChanged(AppTheme value)
    {
        // Applied live so the user sees the switch immediately
        _theme.Apply(value);
    }

    partial void OnApiPortChanged(int value)
    {
        // Applied to the next server start
        _apiHost.RequestedPort = value;
    }

    [RelayCommand]
    private void Back()
    {
        _shell.ShowDashboard();
    }

    [RelayCommand]
    private void Save()
    {
        // TODO: persist to appsettings.json
        _toast.Show("Settings saved");
    }
}
