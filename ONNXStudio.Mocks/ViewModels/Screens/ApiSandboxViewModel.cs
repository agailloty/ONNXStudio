using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Mocks.Models;

namespace ONNXStudio.Mocks.ViewModels.Screens;

public partial class ApiSandboxViewModel : ViewModelBase
{
    private readonly MainViewModel _mainViewModel;
    private readonly OnnxModel _model;
    
    [ObservableProperty]
    private ObservableCollection<ApiEndpoint> _endpoints = new();
    
    [ObservableProperty]
    private ApiEndpoint? _selectedEndpoint;
    
    [ObservableProperty]
    private string _requestBody = "{\n  \"input\": {\n    \"data\": [0.1, 0.2, 0.3, 0.4]\n  }\n}";
    
    [ObservableProperty]
    private string _responseBody = string.Empty;
    
    [ObservableProperty]
    private int _statusCode = 200;
    
    [ObservableProperty]
    private long _executionTimeMs = 0;
    
    [ObservableProperty]
    private bool _isRequesting = false;
    
    [ObservableProperty]
    private ObservableCollection<ApiRequestHistory> _history = new();
    
    [ObservableProperty]
    private ApiRequestHistory? _selectedHistoryItem;
    
    [ObservableProperty]
    private string _statusMessage = "Ready";

    [ObservableProperty]
    private string _selectedMethod = "POST";

    public List<string> Methods { get; } = new() { "GET", "POST", "PUT", "DELETE" };

    public ApiSandboxViewModel(MainViewModel mainViewModel, OnnxModel model)
    {
        _mainViewModel = mainViewModel;
        _model = model;
        Title = "ONNX Studio - " + model.Name + " API Sandbox";
        
        // Create endpoints
        Endpoints = new ObservableCollection<ApiEndpoint>(MockDataGenerator.CreateMockApiEndpoints(model));
        SelectedEndpoint = Endpoints.FirstOrDefault(e => e.Method == "POST");
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
    private void SwitchToConfig()
    {
        _mainViewModel.ShowApiConfig(_model);
    }
    
    [RelayCommand]
    private async Task SendRequestAsync()
    {
        if (SelectedEndpoint == null || IsRequesting) return;
        
        IsRequesting = true;
        StatusMessage = "Sending request...";
        
        // Simulate API call delay
        await System.Threading.Tasks.Task.Delay(1000);
        
        // Generate mock response
        var responseBody = GetMockResponse();
        var executionTime = new Random().Next(100, 1000);
        
        StatusCode = 200;
        ResponseBody = responseBody;
        ExecutionTimeMs = executionTime;
        
        // Add to history
        History.Insert(0, new ApiRequestHistory
        {
            EndpointPath = SelectedEndpoint.Path,
            Method = SelectedEndpoint.Method,
            Timestamp = DateTime.Now,
            StatusCode = StatusCode,
            RequestBody = RequestBody,
            ResponseBody = responseBody,
            ExecutionTimeMs = executionTime
        });
        
        // Keep only last 10 items
        if (History.Count > 10)
        {
            History.RemoveAt(History.Count - 1);
        }
        
        StatusMessage = "Request completed in " + executionTime + "ms";
        IsRequesting = false;
    }
    
    [RelayCommand]
    private void CopyAsCurl()
    {
        if (SelectedEndpoint == null) return;
        StatusMessage = "cURL command copied to clipboard";
    }
    
    [RelayCommand]
    private void CopyAsPython()
    {
        if (SelectedEndpoint == null) return;
        StatusMessage = "Python code copied to clipboard";
    }
    
    [RelayCommand]
    private async Task ReRunRequestAsync(ApiRequestHistory historyItem)
    {
        RequestBody = historyItem.RequestBody;
        SelectedHistoryItem = historyItem;
        await SendRequestAsync();
    }
    
    [RelayCommand]
    private void ClearHistory()
    {
        History.Clear();
        StatusMessage = "History cleared";
    }
    
    [RelayCommand]
    private void DeleteHistoryItem(ApiRequestHistory historyItem)
    {
        History.Remove(historyItem);
        StatusMessage = "Request deleted from history";
    }
    
    private string GetMockResponse()
    {
        if (SelectedEndpoint == null) return "{}";
        
        if (SelectedEndpoint.Path.Contains("/models"))
        {
            return "{\n  \"models\": [\n    \"" + _model.Name + "\"\n  ]\n}";
        }
        else if (SelectedEndpoint.Path.Contains("/schema"))
        {
            return "{\n  \"inputs\": [\n    {\n      \"name\": \"input\",\n      \"type\": \"float32\",\n      \"shape\": [1, 3, 224, 224]\n    }\n  ],\n  \"outputs\": [\n    {\n      \"name\": \"output\",\n      \"type\": \"float32\",\n      \"shape\": [1, 1000]\n    }\n  ]\n}";
        }
        else
        {
            return "{\n  \"output\": [0.85, 0.15],\n  \"executionTimeMs\": 450\n}";
        }
    }
    
    public string ModelName => _model.Name;
    public string StatusColor => GetStatusColor();

    public string GetStatusColor()
    {
        return StatusCode switch
        {
            >= 200 and < 300 => "#3FB950",
            >= 400 and < 500 => "#F8E454",
            >= 500 => "#F85149",
            _ => "#8B949E"
        };
    }
    
    public override void OnLoaded()
    {
        base.OnLoaded();
        StatusMessage = "Ready to test API endpoints";
    }
}
