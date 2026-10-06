using System.Collections.ObjectModel;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Api;
using ONNXStudio.Core.Models;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// API configuration (US-005 UI): manage the embedded REST server for a model,
/// and generate the JSON schema / example payload / cURL of its predict
/// endpoint (matching the payload format accepted by PayloadParser).
/// </summary>
public partial class ApiConfigViewModel : ViewModelBase
{
    private readonly MainWindowViewModel _shell;
    private readonly ApiServerHost _host;
    private readonly IToastService _toast;
    private readonly OnnxModel _model;

    [ObservableProperty]
    private int _port;

    [ObservableProperty]
    private bool _isServerRunning;

    [ObservableProperty]
    private string _jsonSchema = string.Empty;

    [ObservableProperty]
    private string _examplePayload = string.Empty;

    [ObservableProperty]
    private string _curlPreview = string.Empty;

    [ObservableProperty]
    private ObservableCollection<string> _sampleValues = new();

    public OnnxModel Model => _model;
    public string EndpointPath => "/models/" + _model.Id + "/predict";
    public string FullEndpointUrl => "POST http://localhost:" + Port + EndpointPath;

    public ApiConfigViewModel(MainWindowViewModel shell, ApiServerHost host, IToastService toast, OnnxModel model)
    {
        _shell = shell;
        _host = host;
        _toast = toast;
        _model = model;
        Title = model.Name + " - API";
        Port = host.RequestedPort;
        IsServerRunning = host.IsRunning;

        // One editable sample value per input (rank-1 arrays default to two values)
        foreach (var input in model.Inputs)
        {
            SampleValues.Add(input.Shape.Count switch
            {
                0 => "1.0",
                1 when input.Shape[0] is long n && n > 1 => "[0.0, 0.0]",
                _ => "[1.0, 2.0, 3.0, 4.0]"
            });
        }

        RegeneratePreviews();
    }

    partial void OnPortChanged(int value)
    {
        _host.RequestedPort = value;
        RegeneratePreviews();
        OnPropertyChanged(nameof(FullEndpointUrl));
    }

    [RelayCommand]
    private void Back()
    {
        _shell.ShowDashboard();
    }

    [RelayCommand]
    private void SwitchToInspector()
    {
        _shell.ShowInspector(_model);
    }

    [RelayCommand]
    private void SwitchToInference()
    {
        _shell.ShowPlayground(_model);
    }

    [RelayCommand]
    private void SwitchToSandbox()
    {
        _shell.ShowApiSandbox(_model);
    }

    [RelayCommand]
    private async Task ToggleServerAsync()
    {
        if (_host.IsRunning)
        {
            await _host.StopAsync().ConfigureAwait(true);
            IsServerRunning = false;
            _toast.Show("API server stopped");
        }
        else
        {
            await _host.StartAsync().ConfigureAwait(true);
            IsServerRunning = true;
            _toast.Show($"API server listening on http://localhost:{_host.Port}");
        }
    }

    private void RegeneratePreviews()
    {
        var schema = new StringBuilder();
        schema.AppendLine("{");
        schema.AppendLine("  \"$schema\": \"http://json-schema.org/draft-07/schema#\",");
        schema.AppendLine("  \"title\": \"" + _model.Name + " predict request\",");
        schema.AppendLine("  \"type\": \"object\",");
        schema.AppendLine("  \"required\": [\"inputs\"],");
        schema.AppendLine("  \"properties\": {");
        schema.AppendLine("    \"inputs\": {");
        schema.AppendLine("      \"type\": \"object\",");
        schema.AppendLine("      \"required\": [" + string.Join(", ", _model.Inputs.Select(i => $"\"{i.Name}\"")) + "],");
        schema.AppendLine("      \"properties\": {");

        var props = _model.Inputs.Select(i =>
            $"        \"{i.Name}\": {{ \"description\": \"{i.Display}\", \"type\": [\"number\", \"array\"] }}");
        schema.AppendLine(string.Join(",\n", props));
        schema.AppendLine("      }");
        schema.AppendLine("    }");
        schema.AppendLine("  }");
        schema.AppendLine("}");
        JsonSchema = schema.ToString();

        var payloadEntries = _model.Inputs.Select((input, i) =>
            "    \"" + input.Name + "\": " + (i < SampleValues.Count && !string.IsNullOrWhiteSpace(SampleValues[i])
                ? SampleValues[i]
                : "[0.0]"));
        ExamplePayload = "{\n  \"inputs\": {\n" + string.Join(",\n", payloadEntries) + "\n  }\n}";

        CurlPreview = "curl -X POST http://localhost:" + Port + EndpointPath +
                      " \\\n  -H \"Content-Type: application/json\" \\\n  -d '" +
                      ExamplePayload.Replace("\n", "") + "'";
    }
}
