using System;
using System.Collections.Generic;

namespace ONNXStudio.Mocks.Models;

/// <summary>
/// Represents an API endpoint for model inference
/// </summary>
public class ApiEndpoint
{
    public string Method { get; set; } = "POST";
    public string Path { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string ModelId { get; set; } = string.Empty;
    public string ModelName { get; set; } = string.Empty;
    
    // Request/Response examples
    public string RequestExample { get; set; } = string.Empty;
    public string ResponseExample { get; set; } = string.Empty;
    
    // For request builder
    public string RequestBody { get; set; } = string.Empty;
    public Dictionary<string, string> Headers { get; set; } = new();
    
    // Response
    public int StatusCode { get; set; } = 200;
    public string ResponseBody { get; set; } = string.Empty;
    public long ExecutionTimeMs { get; set; }
    
    public string MethodColor => Method switch
    {
        "GET" => "#3FB950",
        "POST" => "#58A6FF",
        "PUT" => "#F8E454",
        "DELETE" => "#F85149",
        _ => "#8B949E"
    };
}

/// <summary>
/// Represents API request history item
/// </summary>
public class ApiRequestHistory
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string EndpointPath { get; set; } = string.Empty;
    public string Method { get; set; } = "GET";
    public DateTime Timestamp { get; set; } = DateTime.Now;
    public int StatusCode { get; set; }
    public string RequestBody { get; set; } = string.Empty;
    public string ResponseBody { get; set; } = string.Empty;
    public long ExecutionTimeMs { get; set; }
    
    public string StatusColor => StatusCode switch
    {
        >= 200 and < 300 => "#3FB950",
        >= 400 and < 500 => "#F8E454",
        >= 500 => "#F85149",
        _ => "#8B949E"
    };
    
    public string TimeDisplay => Timestamp.ToString("HH:mm:ss");
}

/// <summary>
/// Represents API sandbox state
/// </summary>
public class ApiSandboxState
{
    public string SelectedEndpointPath { get; set; } = string.Empty;
    public string SelectedMethod { get; set; } = "POST";
    public string RequestBody { get; set; } = string.Empty;
    public bool IsRequesting { get; set; }
    public ApiEndpoint? CurrentResponse { get; set; }
    public List<ApiRequestHistory> History { get; set; } = new();
}
