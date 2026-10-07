using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Microsoft.Extensions.DependencyInjection;
using ONNXStudio.Api;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Services;
using ONNXStudioUI.ViewModels;
using ONNXStudioUI.ViewModels.Screens;
using ONNXStudioUI.Views.Screens;
using Xunit;

namespace ONNXStudio.UI.Tests;

public class WorkflowTests
{
    [AvaloniaFact]
    public async Task LoadInspectInferServeAndUnloadRealModel()
    {
        await using var services = UiTestSetup.Services();
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var coordinator = services.GetRequiredService<IModelLoadCoordinator>();
        await coordinator.LoadAsync(Path.Combine(AppContext.BaseDirectory, "fixtures", "add.onnx"));
        Dispatcher.UIThread.RunJobs();
        Assert.IsType<DashboardViewModel>(shell.CurrentViewModel);
        Assert.True(services.GetRequiredService<ApiServerHost>().IsRunning);
        var model = Assert.Single(shell.Models);
        shell.ShowInspector(model);
        var inspector = Assert.IsType<ModelInspectorViewModel>(shell.CurrentViewModel);
        Assert.NotEmpty(inspector.Nodes);
        inspector.SelectNodeCommand.Execute(inspector.Nodes[0]);
        Assert.True(inspector.ShowNodeDetails);

        shell.ShowPlayground(model);
        var playground = Assert.IsType<InferencePlaygroundViewModel>(shell.CurrentViewModel);
        playground.Fields[0].ValueText = "1,2";
        playground.Fields[1].ValueText = "3,4";
        await playground.RunInferenceCommand.ExecuteAsync(null);
        Assert.Null(playground.Error);
        Assert.Equal(new double[] { 4, 6 }, playground.Outputs[0].TopValues);
        shell.ShowInspector(model);
        shell.ShowPlayground(model);
        Assert.Same(playground, shell.CurrentViewModel);
        playground.Fields[0].ValueText = "invalid";
        await playground.RunInferenceCommand.ExecuteAsync(null);
        Assert.NotNull(playground.Error);
        Assert.False(playground.HasResult);
        Assert.Empty(playground.Outputs);

        var host = services.GetRequiredService<ApiServerHost>();
        host.RequestedPort = 0;
        shell.ShowApiConfig(model);
        var config = Assert.IsType<ApiConfigViewModel>(shell.CurrentViewModel);
        shell.ShowApiSandbox(model);
        var sandbox = Assert.IsType<ApiSandboxViewModel>(shell.CurrentViewModel);
        await sandbox.SendCommand.ExecuteAsync(null);
        Dispatcher.UIThread.RunJobs();
        Assert.Equal(200, sandbox.StatusCode);
        Assert.True(config.IsServerRunning);
        Assert.Contains($":{host.Port}/", config.FullEndpointUrl);
        Assert.NotEmpty(sandbox.ResponseHeaders);
        sandbox.SelectedEndpoint = "GET /health";
        await sandbox.SendCommand.ExecuteAsync(null);
        Assert.Equal(200, sandbox.StatusCode);
        sandbox.SelectedEndpoint = sandbox.Endpoints[0];
        sandbox.RequestBody = "{";
        await sandbox.SendCommand.ExecuteAsync(null);
        Assert.Equal(0, sandbox.StatusCode);
        Assert.Contains("Invalid JSON", sandbox.ResponseBody);
        await config.ToggleServerCommand.ExecuteAsync(null);
        Dispatcher.UIThread.RunJobs();
        Assert.False(config.IsServerRunning);
        await config.ToggleServerCommand.ExecuteAsync(null);
        var stopped = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        void OnStopped() { if (!host.IsRunning) stopped.TrySetResult(); }
        host.StateChanged += OnStopped;
        services.GetRequiredService<IModelRegistry>().Unload(model.Id);
        await stopped.Task.WaitAsync(TimeSpan.FromSeconds(5));
        host.StateChanged -= OnStopped;
        Dispatcher.UIThread.RunJobs();
        Assert.Empty(shell.Models);
        shell.ShowPlayground(model);
        Assert.NotSame(playground, shell.CurrentViewModel);
    }

    [AvaloniaFact]
    public async Task EveryScreenLoadsItsXamlAndBindings()
    {
        await using var services = UiTestSetup.Services();
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var model = await UiTestSetup.Load(services, "convnet.onnx");
        var window = new Window { Width = 1280, Height = 800 };
        try
        {
            shell.ShowWelcome();
            Show(new WelcomeView());
            shell.ShowDashboard();
            Show(new DashboardView());
            shell.ShowInspector(model);
            Show(new ModelInspectorView());
            shell.ShowPlayground(model);
            Show(new InferencePlaygroundView());
            shell.ShowApiConfig(model);
            Show(new ApiConfigView());
            shell.ShowApiSandbox(model);
            Show(new ApiSandboxView());
            shell.ShowSettings();
            Show(new SettingsView());
        }
        finally { window.Close(); }
        void Show(Control view)
        {
            view.DataContext = shell.CurrentViewModel;
            window.Content = view;
            window.Show();
            window.UpdateLayout();
            Dispatcher.UIThread.RunJobs();
            Assert.True(view.Bounds.Width > 0);
        }
    }

    [AvaloniaFact]
    public async Task FailedLoadCanReturnToWelcome()
    {
        await using var services = UiTestSetup.Services();
        var shell = services.GetRequiredService<MainWindowViewModel>();
        await services.GetRequiredService<IModelLoadCoordinator>().LoadAsync("missing.onnx");
        var loading = Assert.IsType<ModelLoadingViewModel>(shell.CurrentViewModel);
        Assert.True(loading.HasError);
        loading.CancelCommand.Execute(null);
        Assert.IsType<WelcomeViewModel>(shell.CurrentViewModel);
    }

    [AvaloniaFact]
    public async Task BatchLoadStopsOnErrorWithoutHidingIt()
    {
        await using var services = UiTestSetup.Services();
        var shell = services.GetRequiredService<MainWindowViewModel>();
        await services.GetRequiredService<IModelLoadCoordinator>().LoadManyAsync(new[]
        {
            "missing.onnx", Path.Combine(AppContext.BaseDirectory, "fixtures", "add.onnx")
        });
        Assert.True(Assert.IsType<ModelLoadingViewModel>(shell.CurrentViewModel).HasError);
        Assert.Empty(services.GetRequiredService<IModelRegistry>().Models);
    }

    [AvaloniaFact]
    public async Task CancellationReachesLoaderAndPreventsRegistration()
    {
        var loader = new WaitingLoader();
        await using var services = UiTestSetup.Services(loader);
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var load = services.GetRequiredService<IModelLoadCoordinator>().LoadAsync("test.onnx");
        await loader.Started.Task;
        Assert.IsType<ModelLoadingViewModel>(shell.CurrentViewModel).CancelCommand.Execute(null);
        await load;
        Assert.True(loader.Cancelled);
        Assert.Empty(services.GetRequiredService<IModelRegistry>().Models);
        Assert.IsType<WelcomeViewModel>(shell.CurrentViewModel);
    }

    private sealed class WaitingLoader : IModelLoader
    {
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool Cancelled { get; private set; }
        public async Task<Result<OnnxModel, ModelLoadError>> LoadAsync(string path, CancellationToken cancellationToken = default)
        {
            Started.TrySetResult();
            try { await Task.Delay(Timeout.Infinite, cancellationToken); }
            catch (OperationCanceledException) { Cancelled = true; throw; }
            throw new InvalidOperationException();
        }
    }
}
