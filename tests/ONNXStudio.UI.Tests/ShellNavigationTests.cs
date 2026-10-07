using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using ONNXStudioUI;
using ONNXStudioUI.ViewModels;
using ONNXStudioUI.ViewModels.Screens;
using ONNXStudioUI.Views.Screens;
using Xunit;

namespace ONNXStudio.UI.Tests;

public class ShellNavigationTests
{
    [AvaloniaFact]
    public async Task LoadingFromWelcomeReplacesTheDisplayedScreenWithDashboard()
    {
        await using var services = UiTestSetup.Services();
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var content = new ContentControl { DataContext = shell, ContentTemplate = new ViewLocator() };
        content.Bind(ContentControl.ContentProperty, new Binding(nameof(shell.CurrentViewModel)));
        var window = new Window { Width = 1280, Height = 800, Content = content };
        try
        {
            shell.ShowWelcome();
            window.Show();
            window.UpdateLayout();
            Assert.Single(content.GetVisualDescendants().OfType<WelcomeView>());

            await services.GetRequiredService<IModelLoadCoordinator>().LoadAsync(
                Path.Combine(AppContext.BaseDirectory, "fixtures", "add.onnx"));
            Dispatcher.UIThread.RunJobs();
            window.UpdateLayout();

            Assert.IsType<DashboardViewModel>(shell.CurrentViewModel);
            Assert.Empty(content.GetVisualDescendants().OfType<WelcomeView>());
            var dashboard = Assert.Single(content.GetVisualDescendants().OfType<DashboardView>());
            Assert.Same(shell.CurrentViewModel, dashboard.DataContext);
            Assert.Contains(dashboard.GetVisualDescendants().OfType<TextBlock>(), t => t.Text == "add");

            var inspect = Assert.Single(dashboard.GetVisualDescendants().OfType<Button>(), b => Equals(b.Content, "Inspect"));
            Assert.True(inspect.Bounds.Width > 0 && inspect.Bounds.Height > 0);
            Assert.NotNull(inspect.Command);
            Assert.True(inspect.Command.CanExecute(inspect.CommandParameter));
            inspect.Command.Execute(inspect.CommandParameter);
            Dispatcher.UIThread.RunJobs();
            window.UpdateLayout();
            var inspector = Assert.Single(content.GetVisualDescendants().OfType<ModelInspectorView>());
            Assert.Equal("add", Assert.IsType<ModelInspectorViewModel>(inspector.DataContext).Model.Name);

            // Loading another model must display both cards and their actions.
            await services.GetRequiredService<IModelLoadCoordinator>().LoadAsync(
                Path.Combine(AppContext.BaseDirectory, "fixtures", "linreg.onnx"));
            Dispatcher.UIThread.RunJobs();
            window.UpdateLayout();
            dashboard = Assert.Single(content.GetVisualDescendants().OfType<DashboardView>());
            Assert.Contains(dashboard.GetVisualDescendants().OfType<TextBlock>(), t => t.Text == "add");
            Assert.Contains(dashboard.GetVisualDescendants().OfType<TextBlock>(), t => t.Text == "linreg");
            Assert.Equal(2, dashboard.GetVisualDescendants().OfType<Button>().Count(b => Equals(b.Content, "Inspect")));
        }
        finally { window.Close(); }
    }
}
