using System;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Controls.Templates;
using ONNXStudioUI.ViewModels;
using ONNXStudioUI.Views.Screens;

namespace ONNXStudioUI;

/// <summary>
/// Maps view models to views with a compile-time switch (no reflection: AOT friendly).
/// </summary>
public class ViewLocator : IDataTemplate
{
    public bool SupportsRecycling => false;

    public Control Build(object? param)
    {
        return param switch
        {
            ViewModels.Screens.WelcomeViewModel => new Views.Screens.WelcomeView(),
            ViewModels.Screens.DashboardViewModel => new Views.Screens.DashboardView(),
            ViewModels.Screens.ModelLoadingViewModel => new Views.Screens.ModelLoadingView(),
            ViewModels.Screens.ModelInspectorViewModel => new Views.Screens.ModelInspectorView(),
            ViewModels.Screens.InferencePlaygroundViewModel => new Views.Screens.InferencePlaygroundView(),
            ViewModels.Screens.ApiConfigViewModel => new Views.Screens.ApiConfigView(),
            ViewModels.Screens.ApiSandboxViewModel => new Views.Screens.ApiSandboxView(),
            ViewModels.Screens.SettingsViewModel => new Views.Screens.SettingsView(),
            _ => new TextBlock { Text = "No view for " + param?.GetType().Name }
        };
    }

    public bool Match(object? data)
    {
        return data is ViewModelBase;
    }
}
