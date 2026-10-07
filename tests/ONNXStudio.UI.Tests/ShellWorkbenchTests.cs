using Avalonia.Headless.XUnit;
using Microsoft.Extensions.DependencyInjection;
using ONNXStudioUI.ViewModels;
using ONNXStudioUI.ViewModels.Screens;
using Xunit;

namespace ONNXStudio.UI.Tests;

public class ShellWorkbenchTests
{
    [AvaloniaFact]
    public async Task OpeningScreensCreatesOneTabPerScreenAndSyncsTheExplorer()
    {
        await using var services = UiTestSetup.Services();
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var model = await UiTestSetup.Load(services);
        services.GetRequiredService<ONNXStudio.Core.Services.IModelRegistry>().Register(model);
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        var node = Assert.Single(shell.ExplorerNodes);
        Assert.Equal(4, node.Children.Count);

        shell.ShowWelcome();
        Assert.Empty(shell.Tabs);

        shell.ShowDashboard();
        shell.ShowInspector(model);
        shell.ShowInspector(model);
        Assert.Equal(new[] { "dashboard", "inspector:" + model.Id }, shell.Tabs.Select(t => t.Key));
        Assert.Equal("inspector:" + model.Id, Assert.Single(shell.Tabs, t => t.IsActive).Key);
        Assert.Equal("inspector:" + model.Id, shell.SelectedExplorerNode?.Key);
        Assert.True(node.IsExpanded);

        // Selecting an explorer leaf opens the matching screen.
        shell.SelectedExplorerNode = node.Children.Single(c => c.Key.StartsWith("playground:"));
        Assert.IsType<InferencePlaygroundViewModel>(shell.CurrentViewModel);
        Assert.Equal(3, shell.Tabs.Count);
    }

    [AvaloniaFact]
    public async Task ClosingTabsActivatesTheNeighbourAndFallsBackToWelcome()
    {
        await using var services = UiTestSetup.Services();
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var model = await UiTestSetup.Load(services);
        services.GetRequiredService<ONNXStudio.Core.Services.IModelRegistry>().Register(model);
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        shell.ShowDashboard();
        shell.ShowInspector(model);
        shell.ShowPlayground(model);

        shell.CloseActiveTabCommand.Execute(null);
        Assert.IsType<ModelInspectorViewModel>(shell.CurrentViewModel);
        Assert.Equal(2, shell.Tabs.Count);

        // Re-activating a closed screen restores its cached state.
        shell.ShowPlayground(model);
        Assert.Equal(3, shell.Tabs.Count);

        shell.ActivateTabCommand.Execute(shell.Tabs[0]);
        Assert.IsType<DashboardViewModel>(shell.CurrentViewModel);

        while (shell.Tabs.Count > 0) shell.CloseActiveTabCommand.Execute(null);
        Assert.IsType<WelcomeViewModel>(shell.CurrentViewModel);
        Assert.Null(shell.SelectedExplorerNode);
    }

    [AvaloniaFact]
    public async Task UnloadingAModelRemovesItsTabsAndExplorerNode()
    {
        await using var services = UiTestSetup.Services();
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var model = await UiTestSetup.Load(services);
        services.GetRequiredService<ONNXStudio.Core.Services.IModelRegistry>().Register(model);
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        shell.ShowInspector(model);
        shell.UnloadModelCommand.Execute(model);
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        Assert.Empty(shell.ExplorerNodes);
        Assert.DoesNotContain(shell.Tabs, t => t.Key.EndsWith(model.Id));
        Assert.IsType<DashboardViewModel>(shell.CurrentViewModel);
        Assert.False(shell.HasModels);
    }
}
