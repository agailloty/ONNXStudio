using Avalonia;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Microsoft.Extensions.DependencyInjection;
using ONNXStudio.Api;
using ONNXStudio.Core;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Services;
using ONNXStudioUI.ViewModels;
using ONNXStudioUI.ViewModels.Screens;

[assembly: AvaloniaTestApplication(typeof(ONNXStudio.UI.Tests.UiTestSetup))]

namespace ONNXStudio.UI.Tests;

public sealed class TestApplication : ONNXStudioUI.App
{
    public override void OnFrameworkInitializationCompleted() { }
}

public static class UiTestSetup
{
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<TestApplication>()
        .UseHeadless(new AvaloniaHeadlessPlatformOptions());

    public static ServiceProvider Services(IModelLoader? loader = null, Action<IServiceCollection>? configure = null)
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddOnnxStudioCore();
        services.AddOnnxStudioApi();
        services.AddSingleton<IToastService, TestToast>();
        services.AddSingleton<IThemeService, ThemeService>();
        services.AddSingleton<IFilePickerService, TestPicker>();
        services.AddSingleton<SettingsStore>();
        services.AddSingleton<MainWindowViewModel>();
        services.AddSingleton<IModelLoadCoordinator, ModelLoadCoordinator>();
        services.AddTransient<WelcomeViewModel>();
        services.AddTransient<DashboardViewModel>();
        if (loader != null) services.AddSingleton(loader);
        configure?.Invoke(services);
        var provider = services.BuildServiceProvider();
        provider.GetRequiredService<ApiServerHost>().RequestedPort = 0;
        return provider;
    }

    public static async Task<OnnxModel> Load(IServiceProvider services, string name = "add.onnx")
    {
        var result = await services.GetRequiredService<IModelLoader>().LoadAsync(Path.Combine(AppContext.BaseDirectory, "fixtures", name));
        Xunit.Assert.True(result.IsSuccess, result.Error?.Message);
        return result.Value!;
    }

    public static OnnxModel Model(params TensorSchema[] inputs) => new("test", "test.onnx", 0, "test", "", 18, "1", "", 8,
        new ComputationGraph([], []), inputs, []);
}

public sealed class TestToast : IToastService
{
    public string? Current { get; private set; }
    public event Action? ToastChanged;
    public void Show(string message) { Current = message; ToastChanged?.Invoke(); }
}

public sealed class TestPicker : IFilePickerService
{
    public Task<string?> PickModelFileAsync() => Task.FromResult<string?>(null);
    public Task<string[]> PickModelFilesAsync() => Task.FromResult(Array.Empty<string>());
    public Task<string?> PickImageFileAsync() => Task.FromResult<string?>(null);
    public Task<string?> PickPythonInterpreterAsync() => Task.FromResult<string?>(null);
    public Task<string?> PickFolderAsync(string title) => Task.FromResult<string?>(null);
    public Task<string?> PickOnnxSavePathAsync(string suggestedFileName, string? initialDirectory) => Task.FromResult<string?>(null);
}
