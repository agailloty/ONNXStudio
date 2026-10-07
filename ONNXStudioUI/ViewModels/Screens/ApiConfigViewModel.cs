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
public partial class ApiConfigViewModel : ViewModelBase, IDisposable
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

    [ObservableProperty] private string? _error;

    public OnnxModel Model => _model;
    public string EndpointPath => "/models/" + _model.Id + "/predict";
    public string FullEndpointUrl => "POST http://localhost:" + (_host.IsRunning ? _host.Port : Port) + EndpointPath;

    public ApiConfigViewModel(MainWindowViewModel shell, ApiServerHost host, IToastService toast, OnnxModel model)
    {
        _shell = shell;
        _host = host;
        _toast = toast;
        _model = model;
        Title = model.Name + " - API";
        Port = host.RequestedPort;
        IsServerRunning = host.IsRunning;

        _host.StateChanged += OnHostChanged;
        RegeneratePreviews();
    }

    partial void OnPortChanged(int value)
    {
        if (value is < 0 or > 65535) { Error = "Port must be between 0 and 65535."; return; }
        Error = null;
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
        try
        {
            if (Port is < 0 or > 65535) { Error = "Port must be between 0 and 65535."; return; }
            Error = null;
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
        catch (Exception) { Error = "The API server could not start or stop. Check whether the port is already in use."; }
    }

    private void OnHostChanged()
    {
        Avalonia.Threading.Dispatcher.UIThread.Post(() =>
        {
            IsServerRunning = _host.IsRunning;
            Port = _host.RequestedPort;
            OnPropertyChanged(nameof(FullEndpointUrl));
            RegeneratePreviews();
        });
    }

    private void RegeneratePreviews()
    {
        JsonSchema = ApiExamples.Schema(_model);
        try { ExamplePayload = ApiExamples.Payload(_model); }
        catch (InvalidOperationException ex) { Error = ex.Message; ExamplePayload = ""; }
        CurlPreview = ApiExamples.Curl("POST", FullEndpointUrl[5..], ExamplePayload);
    }

    public void Dispose() => _host.StateChanged -= OnHostChanged;
}
