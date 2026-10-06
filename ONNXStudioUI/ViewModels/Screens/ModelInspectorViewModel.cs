using System;
using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Core.Models;
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
/// Key/value pair for the node attribute table.
/// </summary>
public sealed class AttributeEntry
{
    public AttributeEntry(string key, string value)
    {
        Key = key;
        Value = value;
    }

    public string Key { get; }
    public string Value { get; }
}

/// <summary>
/// Model inspector (S-04 / US-002): interactive graph with search and filter,
/// node details sidebar, model statistics and weight list.
/// </summary>
public partial class ModelInspectorViewModel : ViewModelBase
{
    private readonly MainWindowViewModel _shell;
    private readonly IGraphAnalysisService _graphService;
    private readonly OnnxModel _model;

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

    public OnnxModel Model => _model;
    public GraphStatistics Statistics { get; }
    public IReadOnlyList<string> Categories { get; }
    public IReadOnlyList<InitializerInfo> Initializers => _model.Initializers;

    public ModelInspectorViewModel(MainWindowViewModel shell, IGraphAnalysisService graphService, OnnxModel model)
    {
        _shell = shell;
        _graphService = graphService;
        _model = model;
        Title = model.Name + " - Inspector";

        Statistics = graphService.GetStatistics(model);
        Categories = graphService.GetCategories(model);
        RefreshNodes();
    }

    partial void OnSearchTextChanged(string value) => RefreshNodes();
    partial void OnSelectedCategoryChanged(string value) => RefreshNodes();

    private void RefreshNodes()
    {
        var results = _graphService.Search(_model, SearchText, SelectedCategory);
        Nodes = new ObservableCollection<NodeItemViewModel>(
            results.Select(n => new NodeItemViewModel(
                n,
                _graphService.GetCategory(n.OpType),
                CategoryBrushes.For(_graphService.GetCategory(n.OpType)))));
    }

    [RelayCommand]
    private void SelectNode(NodeItemViewModel node)
    {
        SelectedNode = node;
        ShowNodeDetails = true;

        SelectedNodeAttributes = new ObservableCollection<AttributeEntry>(
            node.Node.Attributes.Select(a => new AttributeEntry(a.Key, FormatValue(a.Value))));

        SelectedNodeInputs = new ObservableCollection<string>(node.Node.InputIds);
        SelectedNodeOutputs = new ObservableCollection<string>(node.Node.OutputIds);

        var deps = _graphService.GetDependencies(_model, node.Node.Id);
        DependencySummary = string.Format(
            "depends on {0} node(s) - used by {1} node(s)",
            deps.DependsOn.Count, deps.DependedBy.Count);
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

    private static string FormatValue(object value) => value switch
    {
        List<long> longs => "[" + string.Join(", ", longs) + "]",
        List<float> floats => "[" + string.Join(", ", floats) + "]",
        float f => f.ToString("0.###"),
        _ => value.ToString() ?? string.Empty
    };
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
            "Normalization" => "NormalizationBrush",
            "Pipeline" => "PoolBrush",
            _ => "OtherBrush"
        };

        return (Avalonia.Application.Current?.TryGetResource(key, Avalonia.Styling.ThemeVariant.Default, out var brush)
            == true ? brush : null) as Avalonia.Media.IBrush
            ?? Avalonia.Media.Brushes.Gray;
    }
}
