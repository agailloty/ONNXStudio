using Avalonia;
using Avalonia.Styling;
using ONNXStudio.Core.Models;

namespace ONNXStudioUI.Services;

public enum AppTheme
{
    Light,
    Dark,
    System
}

/// <summary>
/// Applies the requested theme to the whole application (FluentTheme + custom palettes).
/// </summary>
public interface IThemeService
{
    AppTheme Current { get; }
    void Apply(AppTheme theme);
}

public sealed class ThemeService : IThemeService
{
    private AppTheme _current = AppTheme.Light;

    public AppTheme Current => _current;

    public void Apply(AppTheme theme)
    {
        _current = theme;

        if (Application.Current is { } app)
        {
            app.RequestedThemeVariant = theme switch
            {
                AppTheme.Dark => ThemeVariant.Dark,
                AppTheme.Light => ThemeVariant.Light,
                _ => null // follow the system
            };
        }
    }
}
