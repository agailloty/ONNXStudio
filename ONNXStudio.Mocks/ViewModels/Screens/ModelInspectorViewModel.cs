using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Mocks.Models;

namespace ONNXStudio.Mocks.ViewModels.Screens;

public partial class ModelInspectorViewModel : ViewModelBase
{
    private readonly MainViewModel _mainViewModel;
    private readonly OnnxModel _model;
    
    [ObservableProperty]
    private GraphNode? _selectedNode;
    
    [ObservableProperty]
    private string _searchText = string.Empty;
    
    [ObservableProperty]
    private string _filterType = "All";
    
    [ObservableProperty]
    private ObservableCollection<GraphNode> _filteredNodes = new();
    
    [ObservableProperty]
    private double _zoomLevel = 1.0;
    
    [ObservableProperty]
    private double _panX = 0;
    
    [ObservableProperty]
    private double _panY = 0;
    
    [ObservableProperty]
    private bool _showNodeDetails = false;

    [ObservableProperty]
    private string _viewMode = "Graph"; // "Graph" | "Structure"

    [ObservableProperty]
    private bool _showGraph = true;

    [ObservableProperty]
    private bool _showStructure = false;

    [ObservableProperty]
    private ObservableCollection<PipelineComponent> _components = new();

    [ObservableProperty]
    private PipelineComponent? _selectedComponent;

    public ComputationGraph Graph => _model.Graph;
    public OnnxModel Model => _model;
    public string MetadataDisplay => GetMetadataDisplay();
    public List<FeatureImportance> FeatureImportances { get; } = new();
    public bool HasFeatureImportances => FeatureImportances.Count > 0;

    public string StructureSummary => _model.Name.Contains("housing")
        ? "California Housing (20,640 block groups) - R\u00B2 (test): 0.80 - RMSE: 0.50 - target: MedHouseVal"
        : "Producer: " + _model.Producer + " - Opset: " + _model.Graph.OpsetVersion + " - Parameters: " + _model.Graph.ParameterCount.ToString("N0");

    public List<string> FilterTypes { get; } = new() { "All", "Conv", "Pool", "Activation", "Linear", "Normalization", "Other" };
    
    public ModelInspectorViewModel(MainViewModel mainViewModel, OnnxModel model)
    {
        _mainViewModel = mainViewModel;
        _model = model;
        Title = "ONNX Studio - " + model.Name + " Inspector";

        FilteredNodes = new ObservableCollection<GraphNode>(model.Graph.Nodes);

        // Structure (graph-free exploration) data
        Components = new ObservableCollection<PipelineComponent>(
            MockDataGenerator.CreateMockPipelineStructure(model));
        SelectedComponent = Components.FirstOrDefault();
        FeatureImportances.AddRange(MockDataGenerator.CreateMockFeatureImportances(model));
    }

    [RelayCommand]
    private void ShowGraphView()
    {
        ViewMode = "Graph";
        ShowGraph = true;
        ShowStructure = false;
    }

    [RelayCommand]
    private void ShowStructureView()
    {
        ViewMode = "Structure";
        ShowGraph = false;
        ShowStructure = true;
    }
    
    [RelayCommand]
    private void Back()
    {
        _mainViewModel.ShowDashboard();
    }
    
    [RelayCommand]
    private void SwitchToInference()
    {
        _mainViewModel.ShowInferencePlayground(_model);
    }
    
    [RelayCommand]
    private void SwitchToApi()
    {
        _mainViewModel.ShowApiConfig(_model);
    }
    
    [RelayCommand]
    private void SelectNode(GraphNode node)
    {
        SelectedNode = node;
        ShowNodeDetails = true;
    }
    
    [RelayCommand]
    private void SearchNodes()
    {
        UpdateFilteredNodes();
    }
    
    [RelayCommand]
    private void FilterNodes()
    {
        UpdateFilteredNodes();
    }
    
    [RelayCommand]
    private void ResetView()
    {
        ZoomLevel = 1.0;
        PanX = 0;
        PanY = 0;
    }
    
    [RelayCommand]
    private void ZoomIn()
    {
        ZoomLevel *= 1.2;
    }
    
    [RelayCommand]
    private void ZoomOut()
    {
        ZoomLevel /= 1.2;
    }
    
    [RelayCommand]
    private void FitToView()
    {
        ZoomLevel = 1.0;
        PanX = 0;
        PanY = 0;
    }
    
    private void UpdateFilteredNodes()
    {
        var nodes = _model.Graph.Nodes.AsEnumerable();
        
        // Filter by search text
        if (!string.IsNullOrWhiteSpace(SearchText))
        {
            nodes = nodes.Where(n => 
                n.Name.Contains(SearchText, StringComparison.OrdinalIgnoreCase) ||
                n.OpType.Contains(SearchText, StringComparison.OrdinalIgnoreCase)
            );
        }
        
        // Filter by type
        if (FilterType != "All")
        {
            nodes = nodes.Where(n => 
            {
                if (FilterType == "Conv") return n.OpType.Contains("Conv");
                if (FilterType == "Pool") return n.OpType.Contains("Pool");
                if (FilterType == "Activation") return n.OpType.Contains("Relu") || n.OpType.Contains("Sigmoid") || n.OpType.Contains("Tanh");
                if (FilterType == "Linear") return n.OpType.Contains("Gemm") || n.OpType.Contains("MatMul");
                if (FilterType == "Normalization") return n.OpType.Contains("Norm");
                return false; // Other
            });
        }
        
        FilteredNodes = new ObservableCollection<GraphNode>(nodes);
    }
    
    public string GetMetadataDisplay()
    {
        return $"Producer: {_model.Producer}, Opset: {_model.Graph.OpsetVersion}, " +
               $"Nodes: {_model.Graph.NodeCount}, " +
               $"Params: {_model.Graph.ParameterCount:N0}, " +
               $"Inputs: {_model.Graph.Inputs.Count}, " +
               $"Outputs: {_model.Graph.Outputs.Count}";
    }
    
    public override void OnLoaded()
    {
        base.OnLoaded();
        SelectedNode = null;
        ShowNodeDetails = false;
    }
}
