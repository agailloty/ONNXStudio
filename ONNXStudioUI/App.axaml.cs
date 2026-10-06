using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using ONNXStudio.Api;
using ONNXStudio.Core;
using ONNXStudioUI.Services;
using ONNXStudioUI.ViewModels;

namespace ONNXStudioUI;

public partial class App : Application
{
    /// <summary>
    /// Root service provider of the application (modular monolith DI container).
    /// </summary>
    public static IServiceProvider Services { get; private set; } = null!;

    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        Services = BuildServices();

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            // Resolve the shell BEFORE triggering navigation: screen view models
            // depend on it (breaking the DI construction cycle).
            var shell = Services.GetRequiredService<MainWindowViewModel>();
            var window = new Views.MainWindow { DataContext = shell };
            desktop.MainWindow = window;
            shell.ShowWelcome();

            desktop.Exit += (_, _) =>
            {
                if (Services is ServiceProvider provider)
                {
                    provider.Dispose();
                }
            };
        }

        base.OnFrameworkInitializationCompleted();
    }

    private static ServiceProvider BuildServices()
    {
        var services = new ServiceCollection();

        services.AddLogging();
        services.AddOnnxStudioCore();
        services.AddOnnxStudioApi();

        // UI services
        services.AddSingleton<IThemeService, ThemeService>();
        services.AddSingleton<IToastService, ToastService>();
        services.AddSingleton<FilePickerService>();
        services.AddSingleton<IFilePickerService>(sp => sp.GetRequiredService<FilePickerService>());

        // View models
        services.AddSingleton<MainWindowViewModel>();
        services.AddTransient<ViewModels.Screens.WelcomeViewModel>();
        services.AddTransient<ViewModels.Screens.DashboardViewModel>();

        return services.BuildServiceProvider();
    }
}
