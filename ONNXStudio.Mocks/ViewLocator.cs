using System;
using Avalonia.Controls;
using Avalonia.Controls.Templates;
using ONNXStudio.Mocks.ViewModels;
using ONNXStudio.Mocks.ViewModels.Screens;
using ONNXStudio.Mocks.Views.Screens;

namespace ONNXStudio.Mocks;

public class ViewLocator : IDataTemplate
{
    public bool SupportsRecycling => false;

    public Control Build(object? data)
    {
        if (data == null) return new TextBlock { Text = "No view found" };
        
        var name = data.GetType().FullName?.Replace("ViewModel", "View")
            .Replace("ONNXStudio.Mocks.ViewModels.", "ONNXStudio.Mocks.Views.");

        var type = name != null ? Type.GetType(name) : null;

        if (type != null)
        {
            return (Control)Activator.CreateInstance(type)!;
        }
        
        // Fallback for specific view models
        return data switch
        {
            WelcomeViewModel => new WelcomeView(),
            DashboardViewModel => new DashboardView(),
            ModelLoadingViewModel => new ModelLoadingView(),
            ModelInspectorViewModel => new ModelInspectorView(),
            InferencePlaygroundViewModel => new InferencePlaygroundView(),
            ApiConfigViewModel => new ApiConfigView(),
            ApiSandboxViewModel => new ApiSandboxView(),
            SettingsViewModel => new SettingsView(),
            _ => new TextBlock { Text = "View not found: " + name }
        };
    }

    public bool Match(object? data)
    {
        return data is ViewModelBase;
    }
}
