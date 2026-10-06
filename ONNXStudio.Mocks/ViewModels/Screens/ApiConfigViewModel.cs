using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Linq;
using System.Text;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Mocks.Models;

namespace ONNXStudio.Mocks.ViewModels.Screens;

/// <summary>
/// S-08 API Configuration: expose the loaded model as a REST endpoint and
/// define the JSON payload schema by mapping payload fields to model inputs.
/// </summary>
public partial class ApiConfigViewModel : ViewModelBase
{
    private readonly MainViewModel _mainViewModel;
    private readonly OnnxModel _model;

    [ObservableProperty]
    private ObservableCollection<ApiFieldMapping> _mappings = new();

    [ObservableProperty]
    private string _endpointPath = string.Empty;

    [ObservableProperty]
    private int _apiPort = 5000;

    [ObservableProperty]
    private bool _isEndpointEnabled = true;

    [ObservableProperty]
    private string _jsonSchema = string.Empty;

    [ObservableProperty]
    private string _examplePayload = string.Empty;

    [ObservableProperty]
    private string _curlPreview = string.Empty;

    [ObservableProperty]
    private string _statusMessage = "Configure the payload schema, then test in the sandbox";

    public string ModelName => _model.Name;
    public ComputationGraph Graph => _model.Graph;
    public List<string> JsonTypes { get; } = new() { "number", "array", "string", "boolean" };

    public ApiConfigViewModel(MainViewModel mainViewModel, OnnxModel model)
    {
        _mainViewModel = mainViewModel;
        _model = model;
        Title = "ONNX Studio - " + model.Name + " API Configuration";

        EndpointPath = "/models/" + model.Name.ToLowerInvariant() + "/predict";
        ApiPort = mainViewModel.Settings.ApiPort;

        foreach (var mapping in MockDataGenerator.CreateMockApiFieldMappings(model))
        {
            mapping.PropertyChanged += OnMappingChanged;
            Mappings.Add(mapping);
        }

        Mappings.CollectionChanged += OnMappingsChanged;

        RegeneratePreviews();
    }

    private void OnMappingsChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (e.NewItems != null)
        {
            foreach (ApiFieldMapping item in e.NewItems)
            {
                item.PropertyChanged += OnMappingChanged;
            }
        }
        if (e.OldItems != null)
        {
            foreach (ApiFieldMapping item in e.OldItems)
            {
                item.PropertyChanged -= OnMappingChanged;
            }
        }
        RegeneratePreviews();
    }

    private void OnMappingChanged(object? sender, PropertyChangedEventArgs e)
    {
        RegeneratePreviews();
    }

    [RelayCommand]
    private void Back()
    {
        _mainViewModel.ShowDashboard();
    }

    [RelayCommand]
    private void SwitchToInspector()
    {
        _mainViewModel.ShowModelInspector(_model);
    }

    [RelayCommand]
    private void SwitchToInference()
    {
        _mainViewModel.ShowInferencePlayground(_model);
    }

    [RelayCommand]
    private void SwitchToSandbox()
    {
        _mainViewModel.ShowApiSandbox(_model);
    }

    [RelayCommand]
    private void AddMapping()
    {
        Mappings.Add(new ApiFieldMapping
        {
            SourceName = _model.Graph.Inputs.Count > 0 ? _model.Graph.Inputs[0].Name : "input",
            SourceType = _model.Graph.Inputs.Count > 0 ? _model.Graph.Inputs[0].DataType : "float32",
            JsonName = "new_field",
            JsonType = "number",
            IsRequired = false,
            SampleValue = "0.0",
            Description = "Custom field"
        });
        StatusMessage = "Mapping added - it is not bound to a model input yet";
    }

    [RelayCommand]
    private void RemoveMapping(ApiFieldMapping mapping)
    {
        Mappings.Remove(mapping);
        StatusMessage = "Mapping removed: " + mapping.JsonName;
    }

    [RelayCommand]
    private void SaveConfiguration()
    {
        _mainViewModel.Settings.ApiPort = ApiPort;
        _mainViewModel.SaveSettings();
        StatusMessage = "Endpoint configuration saved - serving " + EndpointPath
                        + " on port " + ApiPort;
        _mainViewModel.ShowToast("Endpoint " + EndpointPath + " saved (port " + ApiPort + ")");
    }

    partial void OnEndpointPathChanged(string value) => RegeneratePreviews();
    partial void OnApiPortChanged(int value) => RegeneratePreviews();

    /// <summary>
    /// Rebuilds the JSON Schema, the example payload and the cURL preview
    /// from the current mappings. Called on every change so the preview is live.
    /// </summary>
    private void RegeneratePreviews()
    {
        var active = Mappings
            .Where(m => !string.IsNullOrWhiteSpace(m.JsonName))
            .ToList();

        JsonSchema = BuildJsonSchema(active);
        ExamplePayload = BuildExamplePayload(active);
        CurlPreview = BuildCurl(active);
    }

    private string BuildJsonSchema(List<ApiFieldMapping> mappings)
    {
        var sb = new StringBuilder();
        sb.AppendLine("{");
        sb.AppendLine("  \"$schema\": \"http://json-schema.org/draft-07/schema#\",");
        sb.AppendLine("  \"title\": \"" + _model.Name + " inference request\",");
        sb.AppendLine("  \"type\": \"object\",");
        sb.AppendLine("  \"properties\": {");

        var props = new List<string>();
        foreach (var m in mappings)
        {
            var prop = new List<string> { "    \"" + m.JsonName + "\": {" };
            prop.Add("      \"type\": \"" + m.JsonType + "\"");
            if (!string.IsNullOrWhiteSpace(m.DefaultValue))
            {
                prop.Add("      \"default\": " + m.DefaultValue);
            }
            if (!string.IsNullOrWhiteSpace(m.Description))
            {
                prop.Add("      \"description\": \"" + m.Description + "\"");
            }
            prop.Add("      \"x-model-input\": \"" + m.SourceDisplay + "\"");
            props.Add(string.Join(",\n", prop) + "\n    }");
        }

        sb.AppendLine(string.Join(",\n", props));
        sb.AppendLine("  },");

        var required = mappings.Where(m => m.IsRequired).Select(m => "\"" + m.JsonName + "\"").ToList();
        if (required.Count > 0)
        {
            sb.AppendLine("  \"required\": [" + string.Join(", ", required) + "]");
        }
        else
        {
            sb.AppendLine("  \"required\": []");
        }

        sb.Append("}");
        return sb.ToString();
    }

    private string BuildExamplePayload(List<ApiFieldMapping> mappings)
    {
        var entries = mappings
            .Select(m => "  \"" + m.JsonName + "\": " +
                         (string.IsNullOrWhiteSpace(m.SampleValue) ? m.DefaultValue : m.SampleValue))
            .ToList();

        return "{\n" + string.Join(",\n", entries) + "\n}";
    }

    private string BuildCurl(List<ApiFieldMapping> mappings)
    {
        var payload = BuildExamplePayload(mappings).Replace("\n", "");
        return "curl -X POST http://localhost:" + ApiPort + EndpointPath +
               " \\\n  -H \"Content-Type: application/json\" \\\n  -d '" + payload + "'";
    }
}
