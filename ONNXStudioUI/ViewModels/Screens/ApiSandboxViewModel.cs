using System.Collections.ObjectModel;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Api;
using ONNXStudio.Core.Models;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// One request history entry.
/// </summary>
public partial class HistoryItemViewModel : ObservableObject
{
    public string Endpoint { get; init; } = string.Empty;
    public string TimeDisplay { get; init; } = string.Empty;
    public string RequestBody { get; init; } = string.Empty;
    public string ResponseBody { get; init; } = string.Empty;
    public int StatusCode { get; init; }
    public long ElapsedMs { get; init; }

    public bool IsSuccess => StatusCode is >= 200 and < 300;
}

/// <summary>
/// API sandbox (US-006 UI): send real HTTP requests to the embedded API
/// server and inspect the responses.
/// </summary>
public partial class ApiSandboxViewModel : ViewModelBase, IDisposable
{
    private readonly MainWindowViewModel _shell;
    private readonly ApiServerHost _host;
    private readonly IToastService _toast;
    private readonly OnnxModel _model;
    private readonly HttpClient _httpClient = new() { MaxResponseContentBufferSize = 10 * 1024 * 1024, Timeout = TimeSpan.FromSeconds(30) };

    [ObservableProperty]
    private string _requestBody = string.Empty;

    [ObservableProperty]
    private string _responseBody = string.Empty;

    [ObservableProperty]
    private int _statusCode;

    [ObservableProperty]
    private long _elapsedMs;

    [ObservableProperty]
    private bool _hasResponse;

    [ObservableProperty]
    private bool _isSending;

    [ObservableProperty]
    private ObservableCollection<HistoryItemViewModel> _history = new();

    [ObservableProperty] private string _selectedEndpoint = string.Empty;
    [ObservableProperty] private string _responseHeaders = string.Empty;
    public IReadOnlyList<string> Endpoints { get; }
    public bool RequiresBody => SelectedEndpoint.StartsWith("POST ", StringComparison.Ordinal);
    public string CurlPreview => ApiExamples.Curl(RequiresBody ? "POST" : "GET", EndpointUrl, RequestBody);
    public bool IsSuccess => StatusCode is >= 200 and < 300;
    public Avalonia.Media.IBrush StatusBrush => StatusCode is >= 200 and < 300 ? Avalonia.Media.Brushes.ForestGreen
        : StatusCode is >= 300 and < 400 ? Avalonia.Media.Brushes.DarkOrange : Avalonia.Media.Brushes.IndianRed;
    public OnnxModel Model => _model;
    public string EndpointUrl => $"http://localhost:{(_host.IsRunning ? _host.Port : _host.RequestedPort)}" + SelectedEndpoint[(SelectedEndpoint.IndexOf(' ') + 1)..];
    public bool IsServerRunning => _host.IsRunning;

    public ApiSandboxViewModel(MainWindowViewModel shell, ApiServerHost host, IToastService toast, OnnxModel model)
    {
        _shell = shell;
        _host = host;
        _toast = toast;
        _model = model;
        Title = model.Name + " - Sandbox";

        Endpoints = new[] { $"POST /models/{model.Id}/predict", "GET /models", $"GET /models/{model.Id}", $"GET /models/{model.Id}/schema", "GET /health", "GET /", "GET /openapi.json" };
        SelectedEndpoint = Endpoints[0];
        try { RequestBody = ApiExamples.Payload(model); }
        catch (InvalidOperationException ex) { ResponseBody = ex.Message; HasResponse = true; }
        _host.StateChanged += OnHostChanged;
    }

    partial void OnSelectedEndpointChanged(string value)
    {
        OnPropertyChanged(nameof(EndpointUrl));
        OnPropertyChanged(nameof(RequiresBody));
        OnPropertyChanged(nameof(CurlPreview));
    }
    partial void OnRequestBodyChanged(string value) => OnPropertyChanged(nameof(CurlPreview));
    partial void OnStatusCodeChanged(int value) { OnPropertyChanged(nameof(IsSuccess)); OnPropertyChanged(nameof(StatusBrush)); }
    private void OnHostChanged() => Avalonia.Threading.Dispatcher.UIThread.Post(() =>
    {
        OnPropertyChanged(nameof(IsServerRunning));
        OnPropertyChanged(nameof(EndpointUrl));
        OnPropertyChanged(nameof(CurlPreview));
    });

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
    private void SwitchToConfig()
    {
        _shell.ShowApiConfig(_model);
    }

    [RelayCommand]
    private async Task SendAsync()
    {
        if (IsSending) return;
        IsSending = true;
        StatusCode = 0;
        ElapsedMs = 0;
        HasResponse = false;
        ResponseHeaders = string.Empty;
        var endpoint = SelectedEndpoint;
        var requestBody = RequestBody;
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            if (RequiresBody) { using var json = System.Text.Json.JsonDocument.Parse(requestBody); }
            if (!_host.IsRunning) await _host.StartAsync();
            using var request = new HttpRequestMessage(RequiresBody ? HttpMethod.Post : HttpMethod.Get, EndpointUrl);
            if (RequiresBody) request.Content = new StringContent(requestBody, Encoding.UTF8, "application/json");
            using var response = await _httpClient.SendAsync(request);
            var body = await response.Content.ReadAsStringAsync();
            StatusCode = (int)response.StatusCode;
            ResponseHeaders = string.Join("\n", response.Headers.Concat(response.Content.Headers).Select(h => h.Key + ": " + string.Join(", ", h.Value)));
            try
            {
                var json = System.Text.Json.Nodes.JsonNode.Parse(body);
                ResponseBody = json?.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) ?? body;
            }
            catch (System.Text.Json.JsonException) { ResponseBody = body; }
        }
        catch (System.Text.Json.JsonException)
        {
            ResponseBody = "Invalid JSON. Correct the request body before sending.";
        }
        catch (Exception ex)
        {
            ResponseBody = "Request failed: " + ex.Message;
        }
        finally
        {
            stopwatch.Stop();
            ElapsedMs = stopwatch.ElapsedMilliseconds;
            HasResponse = true;
            IsSending = false;
        }
        History.Insert(0, new HistoryItemViewModel
        {
            Endpoint = endpoint,
            TimeDisplay = DateTime.Now.ToString("HH:mm:ss"),
            RequestBody = requestBody,
            ResponseBody = ResponseBody,
            StatusCode = StatusCode,
            ElapsedMs = ElapsedMs
        });
        foreach (var old in History.Where(h => h.Endpoint == endpoint).Skip(10).ToArray()) History.Remove(old);
        _toast.Show(StatusCode == 0 ? "Request failed" : $"{StatusCode} in {ElapsedMs} ms");
    }

    [RelayCommand]
    private async Task RerunAsync(HistoryItemViewModel item)
    {
        if (IsSending) return;
        SelectedEndpoint = item.Endpoint;
        RequestBody = item.RequestBody;
        await SendAsync();
    }

    public void Dispose()
    {
        _host.StateChanged -= OnHostChanged;
        _httpClient.Dispose();
    }

    [RelayCommand]
    private void ClearHistory()
    {
        History.Clear();
    }
}
