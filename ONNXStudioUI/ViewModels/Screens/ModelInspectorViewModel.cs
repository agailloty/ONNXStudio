using System;
using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Python;
using ONNXStudio.Core.Services;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// UI item for a graph node (with category + brush for color coding).
/// </summary>
public partial class NodeItemViewModel : ObservableObject
{
    public GraphNode Node { get; }
    public string OpType => Node.OpType;
    public string Name => Node.Name;
    public string Category { get; }
    public Avalonia.Media.IBrush CategoryBrush { get; }
    public string Tooltip => Name + " (" + OpType + ")";

    public NodeItemViewModel(GraphNode node, string category, Avalonia.Media.IBrush brush)
    {
        Node = node;
        Category = category;
        CategoryBrush = brush;
    }
}

/// <summary>
/// Model inspector (S-04 / US-002): interactive graph with search and filter,
/// node details sidebar, model statistics and weight list.
/// </summary>
public partial class ModelInspectorViewModel : ViewModelBase
{
    private readonly MainWindowViewModel _shell;
    private readonly IGraphAnalysisService _graphService;
    private readonly IModel _model;
    private readonly Dictionary<string, NodeItemViewModel> _nodeItems;
    private readonly Dictionary<string, (int Inputs, int Outputs)> _dependencyCounts;
    private readonly Lazy<IReadOnlyList<ModelComponent>> _components;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowGraph))]
    private bool _showStructure;

    [ObservableProperty]
    private ModelComponent? _selectedComponent;

    public bool ShowGraph => !ShowStructure;
    public IReadOnlyList<ModelComponent> Components => _components.Value;
    public string StructureSummary => string.Join(" · ", Model.Facts.Select(f => $"{f.Key}: {f.Value}").Append($"{Statistics.NodeCount:N0} nodes"));

    public string WeightsHeader => Model.WeightsLabel;
    public IReadOnlyList<string> Facts => Model.Facts.Select(f => $"{f.Key}: {f.Value}").ToArray();

    [ObservableProperty]
    private ObservableCollection<NodeItemViewModel> _nodes = new();

    [ObservableProperty]
    private string _searchText = string.Empty;

    [ObservableProperty]
    private string _selectedCategory = "All";

    [ObservableProperty]
    private NodeItemViewModel? _selectedNode;

    [ObservableProperty]
    private bool _showNodeDetails;

    [ObservableProperty]
    private ObservableCollection<AttributeEntry> _selectedNodeAttributes = new();

    [ObservableProperty]
    private ObservableCollection<string> _selectedNodeInputs = new();

    [ObservableProperty]
    private ObservableCollection<string> _selectedNodeOutputs = new();

    [ObservableProperty]
    private string _dependencySummary = string.Empty;

    public IModel Model => _model;
    public bool CanExportToOnnx => Model is SklearnModel;

    [RelayCommand(CanExecute = nameof(CanExportToOnnx))]
    private void ExportToOnnx()
    {
        if (Model is SklearnModel sklearn) _shell.ShowPythonModel(sklearn.File);
    }

    public GraphStatistics Statistics { get; }
    public IReadOnlyList<string> Categories { get; }
    public IReadOnlyList<InitializerInfo> Initializers => _model.Initializers;

    public ModelInspectorViewModel(MainWindowViewModel shell, IGraphAnalysisService graphService, IModel model)
    {
        _shell = shell;
        _graphService = graphService;
        _model = model;
        Title = model.Name + " - Inspector";

        Statistics = graphService.GetStatistics(model);
        Categories = graphService.GetCategories(model);
        _nodeItems = model.Graph.Nodes.ToDictionary(n => n.Id, n => new NodeItemViewModel(
            n, graphService.GetCategory(n), CategoryBrushes.For(graphService.GetCategory(n))));
        // Build this once, instead of scanning the entire graph on every click.
        var incoming = new Dictionary<string, HashSet<string>>();
        var outgoing = new Dictionary<string, HashSet<string>>();
        foreach (var edge in model.Graph.Edges)
        {
            if (!_nodeItems.ContainsKey(edge.FromNodeId) || !_nodeItems.ContainsKey(edge.ToNodeId)) continue;
            if (!incoming.TryGetValue(edge.ToNodeId, out var sources)) incoming[edge.ToNodeId] = sources = new();
            if (!outgoing.TryGetValue(edge.FromNodeId, out var targets)) outgoing[edge.FromNodeId] = targets = new();
            sources.Add(edge.FromNodeId);
            targets.Add(edge.ToNodeId);
        }
        _dependencyCounts = _nodeItems.Keys.ToDictionary(id => id,
            id => (incoming.GetValueOrDefault(id)?.Count ?? 0, outgoing.GetValueOrDefault(id)?.Count ?? 0));
        _components = new(() => ModelComponent.Build(model));
        RefreshNodes();
    }

    partial void OnSearchTextChanged(string value) => RefreshNodes();
    partial void OnSelectedCategoryChanged(string value) => RefreshNodes();

    private void RefreshNodes()
    {
        var results = _graphService.Search(_model, SearchText, SelectedCategory);
        Nodes = new ObservableCollection<NodeItemViewModel>(
            results.Select(n => _nodeItems[n.Id]));
    }

    [RelayCommand]
    private void ShowGraphView() => ShowStructure = false;

    [RelayCommand]
    private void ShowStructureView()
    {
        SelectedComponent ??= Components.FirstOrDefault();
        ShowStructure = true;
    }

    partial void OnSelectedComponentChanged(ModelComponent? value)
    {
        if (value?.NodeId is { } id && _nodeItems.TryGetValue(id, out var node)) SelectNode(node);
    }

    [RelayCommand]
    private void SelectNode(NodeItemViewModel node)
    {
        if (ReferenceEquals(SelectedNode, node)) return;
        SelectedNode = node;
        ShowNodeDetails = true;

        SelectedNodeAttributes = new ObservableCollection<AttributeEntry>(
            node.Node.Attributes.Select(a => new AttributeEntry(a.Key, a.Value)));

        SelectedNodeInputs = new ObservableCollection<string>(node.Node.InputIds);
        SelectedNodeOutputs = new ObservableCollection<string>(node.Node.OutputIds);

        var deps = _dependencyCounts[node.Node.Id];
        DependencySummary = string.Format(
            "depends on {0} node(s) - used by {1} node(s)",
            deps.Inputs, deps.Outputs);
    }

    [RelayCommand]
    private void Back()
    {
        _shell.ShowDashboard();
    }

    [RelayCommand]
    private void SwitchToInference()
    {
        _shell.ShowPlayground(_model);
    }

    [RelayCommand]
    private void SwitchToApi()
    {
        _shell.ShowApiConfig(_model);
    }

}

/// <summary>
/// Category color mapping (brushes come from the theme dictionaries).
/// </summary>
public static class CategoryBrushes
{
    public static Avalonia.Media.IBrush For(string category)
    {
        var key = category switch
        {
            "Conv" => "ConvBrush",
            "Pool" => "PoolBrush",
            "Activation" => "ActivationBrush",
            "Linear" => "LinearBrush",
            "Normalization" or "Transformer" => "NormalizationBrush",
            "Classifier" => "LinearBrush",
            "Regressor" => "ActivationBrush",
            "Pipeline" => "PoolBrush",
            _ => "OtherBrush"
        };

        return (Avalonia.Application.Current?.TryGetResource(key, Avalonia.Styling.ThemeVariant.Default, out var brush)
            == true ? brush : null) as Avalonia.Media.IBrush
            ?? Avalonia.Media.Brushes.Gray;
    }
}
