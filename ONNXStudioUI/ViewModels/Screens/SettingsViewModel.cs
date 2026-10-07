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
    private readonly SettingsStore _store;

    [ObservableProperty]
    private AppTheme _selectedTheme;

    [ObservableProperty]
    private int _apiPort;

    public IReadOnlyList<AppTheme> Themes { get; } = new[] { AppTheme.Light, AppTheme.Dark, AppTheme.System };
    public string AppVersion => "1.0.0";

    public SettingsViewModel(MainWindowViewModel shell, IThemeService theme, IToastService toast, ApiServerHost apiHost, SettingsStore store)
    {
        _shell = shell;
        _theme = theme;
        _toast = toast;
        _apiHost = apiHost;
        _store = store;
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
        if (value is >= 0 and <= 65535) _apiHost.RequestedPort = value;
    }

    [RelayCommand]
    private void Back()
    {
        _shell.ShowDashboard();
    }

    [RelayCommand]
    private void Save()
    {
        if (ApiPort is < 0 or > 65535) { _toast.Show("Port must be between 0 and 65535."); return; }
        try
        {
            _store.Save(SelectedTheme, ApiPort);
            _toast.Show("Settings saved");
        }
        catch (Exception ex) when (ex is System.IO.IOException or UnauthorizedAccessException)
        {
            _toast.Show("Settings could not be saved. Check folder permissions.");
        }
    }
}
