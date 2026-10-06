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
public partial class ApiSandboxViewModel : ViewModelBase
{
    private readonly MainWindowViewModel _shell;
    private readonly ApiServerHost _host;
    private readonly IToastService _toast;
    private readonly OnnxModel _model;
    private readonly HttpClient _httpClient = new();

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

    public OnnxModel Model => _model;
    public string EndpointUrl => $"http://localhost:{_host.Port}/models/{_model.Id}/predict";
    public bool IsServerRunning => _host.IsRunning;

    public ApiSandboxViewModel(MainWindowViewModel shell, ApiServerHost host, IToastService toast, OnnxModel model)
    {
        _shell = shell;
        _host = host;
        _toast = toast;
        _model = model;
        Title = model.Name + " - Sandbox";

        // Default request body: one sample value per input
        var entries = model.Inputs.Select(i =>
            "    \"" + i.Name + "\": " + (i.Shape.Count <= 1 ? "1.0" : "[1.0, 2.0, 3.0, 4.0]"));
        RequestBody = "{\n  \"inputs\": {\n" + string.Join(",\n", entries) + "\n  }\n}";
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
    private void SwitchToConfig()
    {
        _shell.ShowApiConfig(_model);
    }

    [RelayCommand]
    private async Task SendAsync()
    {
        if (IsSending)
        {
            return;
        }

        if (!_host.IsRunning)
        {
            await _host.StartAsync().ConfigureAwait(true);
            OnPropertyChanged(nameof(IsServerRunning));
        }

        IsSending = true;
        try
        {
            using var content = new StringContent(RequestBody, Encoding.UTF8, "application/json");
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            using var response = await _httpClient.PostAsync(EndpointUrl, content).ConfigureAwait(true);
            stopwatch.Stop();

            var body = await response.Content.ReadAsStringAsync().ConfigureAwait(true);

            StatusCode = (int)response.StatusCode;
            ResponseBody = body;
            ElapsedMs = stopwatch.ElapsedMilliseconds;
            HasResponse = true;

            History.Insert(0, new HistoryItemViewModel
            {
                TimeDisplay = System.DateTime.Now.ToString("HH:mm:ss"),
                RequestBody = RequestBody,
                ResponseBody = body,
                StatusCode = StatusCode,
                ElapsedMs = ElapsedMs
            });
            while (History.Count > 10)
            {
                History.RemoveAt(History.Count - 1);
            }

            _toast.Show($"{StatusCode} in {ElapsedMs} ms");
        }
        catch (System.Exception ex)
        {
            ResponseBody = "Request failed: " + ex.Message;
            HasResponse = true;
        }
        finally
        {
            IsSending = false;
        }
    }

    [RelayCommand]
    private void Rerun(HistoryItemViewModel item)
    {
        RequestBody = item.RequestBody;
        _ = SendAsync();
    }

    [RelayCommand]
    private void ClearHistory()
    {
        History.Clear();
    }
}
