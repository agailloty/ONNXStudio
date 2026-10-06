using System;
using System.Collections.Generic;

namespace ONNXStudio.Mocks.Models;

/// <summary>
/// Application settings
/// </summary>
public class AppSettings
{
    // General settings
    public ThemeMode Theme { get; set; } = ThemeMode.Light;
    public string Language { get; set; } = "English";
    public StartupAction StartupAction { get; set; } = StartupAction.OpenLastModel;
    
    // API Server settings
    public int ApiPort { get; set; } = 5000;
    public int MaxRequestSizeMb { get; set; } = 10;
    public bool AllowAllOrigins { get; set; } = true;
    
    // Advanced settings
    public LogLevel LogLevel { get; set; } = LogLevel.Info;
    public int CacheSize { get; set; } = 5;
}

public enum ThemeMode
{
    Dark,
    Light,
    System
}

public enum StartupAction
{
    OpenLastModel,
    ShowWelcomeScreen,
    ShowDashboard
}

public enum LogLevel
{
    Debug,
    Info,
    Warning,
    Error
}
