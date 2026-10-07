using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Microsoft.Extensions.DependencyInjection;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Controls;
using ONNXStudioUI.ViewModels;
using ONNXStudioUI.ViewModels.Screens;
using Xunit;

namespace ONNXStudio.UI.Tests;

public class GraphInteractionTests
{
    [AvaloniaFact]
    public async Task SelectZoomPanAndKeyboardNavigationKeepGraphInteractive()
    {
        await using var services = UiTestSetup.Services();
        var model = await UiTestSetup.Load(services, "convnet.onnx");
        var vm = new ModelInspectorViewModel(services.GetRequiredService<MainWindowViewModel>(), services.GetRequiredService<IGraphAnalysisService>(), model);
        var graph = new GraphViewer { Nodes = vm.Nodes, Graph = model.Graph, SelectNodeCommand = vm.SelectNodeCommand };
        var window = new Window { Width = 900, Height = 600, Content = graph };
        try
        {
            window.Show();
            window.UpdateLayout();
            window.MouseDown(new Point(80, 50), MouseButton.Left);
            window.MouseUp(new Point(80, 50), MouseButton.Left);
            Assert.NotNull(vm.SelectedNode);
            var first = vm.SelectedNode;
            window.MouseWheel(new Point(80, 50), new Vector(0, 2));
            window.MouseDown(new Point(80, 50), MouseButton.Left);
            window.MouseUp(new Point(80, 50), MouseButton.Left);
            Assert.Same(first, vm.SelectedNode);
            window.MouseDown(new Point(80, 50), MouseButton.Left);
            window.MouseMove(new Point(180, 150));
            window.MouseUp(new Point(180, 150), MouseButton.Left);
            window.MouseDown(new Point(180, 150), MouseButton.Left);
            window.MouseUp(new Point(180, 150), MouseButton.Left);
            Assert.Same(first, vm.SelectedNode);
            window.KeyPress(Key.Right, RawInputModifiers.None, PhysicalKey.ArrowRight, null);
            Assert.NotSame(first, vm.SelectedNode);
        }
        finally { window.Close(); }
    }
}
