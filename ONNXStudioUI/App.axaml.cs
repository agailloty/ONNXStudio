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

    /// <summary>
    /// Raw command line arguments (set by Program.Main before Avalonia starts).
    /// </summary>
    public static string[] CommandLineArgs { get; set; } = Array.Empty<string>();

    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        Services = BuildServices();
        var settings = Services.GetRequiredService<SettingsStore>().Load();
        Services.GetRequiredService<IThemeService>().Apply(settings.Theme);
        Services.GetRequiredService<ApiServerHost>().RequestedPort = settings.Port;

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
                    provider.DisposeAsync().AsTask().GetAwaiter().GetResult();
                }
            };

            // CLI support: ONNXStudioUI --model path/to/model.onnx
            var modelPath = GetModelArgument();
            if (modelPath != null)
            {
                var coordinator = Services.GetRequiredService<ViewModels.IModelLoadCoordinator>();
                _ = coordinator.LoadAsync(modelPath);
            }
        }

        base.OnFrameworkInitializationCompleted();
    }

    private static string? GetModelArgument()
    {
        var args = CommandLineArgs;
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], "--model", StringComparison.OrdinalIgnoreCase))
            {
                return args[i + 1];
            }
        }
        return args.FirstOrDefault(a => a.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase));
    }

    private static ServiceProvider BuildServices()
    {
        var services = new ServiceCollection();

        services.AddLogging();
        services.AddOnnxStudioCore();
        services.AddOnnxStudioApi();

        // UI services
        services.AddSingleton<IThemeService, ThemeService>();
        services.AddSingleton<SettingsStore>();
        services.AddSingleton<IToastService, ToastService>();
        services.AddSingleton<FilePickerService>();
        services.AddSingleton<IFilePickerService>(sp => sp.GetRequiredService<FilePickerService>());

        // View models
        services.AddSingleton<MainWindowViewModel>();
        services.AddSingleton<ViewModels.IModelLoadCoordinator, ViewModels.ModelLoadCoordinator>();
        services.AddTransient<ViewModels.Screens.WelcomeViewModel>();
        services.AddTransient<ViewModels.Screens.DashboardViewModel>();

        return services.BuildServiceProvider();
    }
}
