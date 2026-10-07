using System.Collections;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Controls;
using ONNXStudioUI.ViewModels;
using ONNXStudioUI.ViewModels.Screens;
using ONNXStudioUI.Views.Screens;
using Xunit;

namespace ONNXStudio.UI.Tests;

public class ModelInspectorTests
{
    [Fact]
    public void LargeAttributeReadsOnlyTheVisiblePage()
    {
        var values = new CountingList(1_000_000);
        var entry = new AttributeEntry("nodes_values", values);
        Assert.True(entry.HasMultiplePages);
        Assert.Equal(0, values.Reads);
        Assert.StartsWith("[0, 1, 2,", entry.Value);
        Assert.Equal(AttributeEntry.PageSize, values.Reads);
        Assert.False(entry.PreviousPageCommand.CanExecute(null));
        entry.NextPageCommand.Execute(null);
        Assert.StartsWith("[16, 17, 18,", entry.Value);
        Assert.Equal(2 * AttributeEntry.PageSize, values.Reads);
        entry.PreviousPageCommand.Execute(null);
        Assert.StartsWith("[0, 1, 2,", entry.Value);
    }

    [Fact]
    public void AttributePagingKeepsAllValuesAndHandlesLastPageAndLongStrings()
    {
        var values = Enumerable.Range(0, 35).Select(i => (long)i).ToList();
        var entry = new AttributeEntry("nodes_nodeids", values);
        entry.NextPageCommand.Execute(null);
        entry.NextPageCommand.Execute(null);
        Assert.Equal("[32, 33, 34]", entry.Value);
        Assert.False(entry.NextPageCommand.CanExecute(null));
        Assert.True(entry.PreviousPageCommand.CanExecute(null));
        Assert.Equal(35, values.Count);
        Assert.Equal("[]", new AttributeEntry("empty", new List<float>()).Value);
        Assert.Equal("[0.125, 1.5]", new AttributeEntry("floats", new List<float> { .125f, 1.5f }).Value);
        Assert.Equal("[BRANCH_LEQ, LEAF]", new AttributeEntry("modes", new[] { "BRANCH_LEQ", "LEAF" }).Value);
        var text = new AttributeEntry("long", new string('a', 1_000_000)).Value;
        Assert.True(text.Length < 300);
        Assert.EndsWith("(truncated)", text);
    }

    [AvaloniaFact]
    public async Task LargeForestCanBeClickedPagedAndInspectedInTheStructureView()
    {
        await using var services = UiTestSetup.Services();
        var values = new CountingList(1_000_000);
        var forest = new GraphNode("forest", "RandomForest", "TreeEnsembleRegressor", "ai.onnx.ml",
            new Dictionary<string, object> { ["nodes_values"] = values, ["nodes_nodeids"] = new long[1_000_000] },
            ["features"], ["prediction"]);
        var next = new GraphNode("next", "Output", "Identity", "", new Dictionary<string, object>(), ["prediction"], ["output"]);
        var model = new OnnxModel("forest", "forest.onnx", 10_000_000, "test producer", "", 18, "1", "", 8,
            new ComputationGraph([forest, next], [new("prediction", "forest", "next")]),
            [new("features", DataType.Float32, [1, 4])], [new("output", DataType.Float32, [1])]);
        var vm = CreateInspector(services, model);
        var view = new ModelInspectorView { DataContext = vm };
        var window = new Window { Width = 1200, Height = 800, Content = view };
        try
        {
            window.Show();
            window.UpdateLayout();
            Assert.Equal(0, values.Reads);
            var graph = view.GetVisualDescendants().OfType<GraphViewer>().Single();
            var point = graph.TranslatePoint(new Point(80, 50), window)!.Value;
            window.MouseDown(point, MouseButton.Left);
            window.MouseUp(point, MouseButton.Left);
            window.UpdateLayout();
            Assert.Same(forest, vm.SelectedNode!.Node);
            Assert.Equal("depends on 0 node(s) - used by 1 node(s)", vm.DependencySummary);
            Assert.True(values.Reads < 200, "Layout must not format the entire forest attribute.");
            var nextPage = view.GetVisualDescendants().OfType<Button>().First(b => Equals(b.Content, "Next"));
            nextPage.Command!.Execute(null);
            window.UpdateLayout();
            Assert.StartsWith("[16, 17,", vm.SelectedNodeAttributes[0].Value);

            var toggle = view.GetVisualDescendants().OfType<Button>().Single(b => Equals(b.Content, "Structure & Parameters"));
            toggle.Command!.Execute(null);
            window.UpdateLayout();
            Assert.True(vm.ShowStructure);
            Assert.False(graph.IsEffectivelyVisible);
            var tree = view.FindControl<TreeView>("StructureTree")!;
            var root = Assert.Single(vm.Components);
            var rootContainer = Assert.IsType<TreeViewItem>(tree.ContainerFromItem(root));
            rootContainer.IsExpanded = true;
            window.UpdateLayout();
            var operators = root.Children[1];
            var operatorContainer = Assert.IsType<TreeViewItem>(rootContainer.ContainerFromItem(operators));
            operatorContainer.IsExpanded = true;
            window.UpdateLayout();
            tree.SelectedItem = operators.Children[0];
            window.UpdateLayout();
            Assert.Same(operators.Children[0], vm.SelectedComponent);
            var attribute = vm.SelectedComponent!.Parameters.Single(p => p.Key == "nodes_values");
            Assert.Equal(1_000_000, attribute.Count);
            Assert.True(attribute.HasMultiplePages);
            tree.SelectedItem = operators.Children[1];
            window.UpdateLayout();
            Assert.Same(next, vm.SelectedNode!.Node);
            vm.ShowGraphViewCommand.Execute(null);
            window.UpdateLayout();
            Assert.True(graph.IsEffectivelyVisible);
            Assert.Same(vm.SelectedNode, graph.SelectedNode);
            Assert.True(values.Reads < 500);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task StructureUsesRealModelMetadataAndKeepsGraphFilters()
    {
        await using var services = UiTestSetup.Services();
        var model = await UiTestSetup.Load(services, "convnet.onnx");
        var vm = CreateInspector(services, model);
        vm.SearchText = model.Graph.Nodes[0].OpType;
        var filtered = vm.Nodes.ToArray();
        vm.ShowStructureViewCommand.Execute(null);
        var root = Assert.Single(vm.Components);
        Assert.Equal(model.Name, root.Header);
        Assert.Equal(model.ProducerName, root.Parameters.Single(p => p.Key == "Producer").Value);
        Assert.Equal(model.Inputs.Count, root.Children[0].Children.Count);
        Assert.Equal(model.Graph.Nodes.Count, root.Children[1].Children.Count);
        Assert.Equal(model.Outputs.Count, root.Children[2].Children.Count);
        Assert.Equal(model.Initializers.Count, root.Children[3].Children.Count);
        vm.ShowGraphViewCommand.Execute(null);
        Assert.Equal(filtered, vm.Nodes);
    }

    [Fact]
    public void HugeStructureHasBoundedBranchesAndEveryNodeRemainsReachable()
    {
        var nodes = Enumerable.Range(0, 10_001).Select(i => new GraphNode(i.ToString(), $"node{i}", "Identity", "",
            new Dictionary<string, object>(), [], [])).ToArray();
        var model = new OnnxModel("large", "large.onnx", 0, "", "", 18, "1", "", 8,
            new ComputationGraph(nodes, []), [], []);
        var operators = ModelComponent.Build(model)[0].Children[1];
        var ids = new List<string>();
        void Visit(ModelComponent component)
        {
            Assert.True(component.Children.Count <= 100);
            if (component.NodeId is { } id) ids.Add(id);
            foreach (var child in component.Children) Visit(child);
        }
        Visit(operators);
        Assert.Equal(nodes.Select(n => n.Id), ids);
        Assert.All(ModelComponent.Build(UiTestSetup.Model())[0].Children, section => Assert.Empty(section.Children));
    }

    private static ModelInspectorViewModel CreateInspector(IServiceProvider services, OnnxModel model) =>
        new(services.GetRequiredService<MainWindowViewModel>(), services.GetRequiredService<IGraphAnalysisService>(), model);

    // Enumeration deliberately fails: previewing must use bounded indexed reads.
    private sealed class CountingList(int count) : IList
    {
        public int Reads { get; private set; }
        public int Count => count;
        public object? this[int index] { get { Reads++; return (float)index; } set => throw new NotSupportedException(); }
        public bool IsFixedSize => true;
        public bool IsReadOnly => true;
        public bool IsSynchronized => false;
        public object SyncRoot => this;
        public IEnumerator GetEnumerator() => throw new InvalidOperationException("Full enumeration freezes the UI.");
        public int Add(object? value) => throw new NotSupportedException();
        public void Clear() => throw new NotSupportedException();
        public bool Contains(object? value) => throw new NotSupportedException();
        public int IndexOf(object? value) => throw new NotSupportedException();
        public void Insert(int index, object? value) => throw new NotSupportedException();
        public void Remove(object? value) => throw new NotSupportedException();
        public void RemoveAt(int index) => throw new NotSupportedException();
        public void CopyTo(Array array, int index) => throw new NotSupportedException();
    }
}
