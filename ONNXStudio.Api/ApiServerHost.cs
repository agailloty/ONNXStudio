using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Cors.Infrastructure;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using ONNXStudio.Api.Endpoints;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Services;

namespace ONNXStudio.Api;

/// <summary>
/// Hosts the embedded REST API inside the desktop application process
/// (modular monolith: ASP.NET Core Kestrel alongside the Avalonia UI).
/// </summary>
public sealed class ApiServerHost : IAsyncDisposable
{
    private readonly IModelRegistry _registry;
    private readonly IInferenceService _inferenceService;
    private readonly OnnxStudioOptions _options;
    private readonly ILogger<ApiServerHost>? _logger;
    private WebApplication? _app;

    public int Port { get; private set; }
    public bool IsRunning => _app != null;

    /// <summary>
    /// Port used on the next start (configurable from the API config screen).
    /// Defaults to the configured options port when not overridden.
    /// </summary>
    public int RequestedPort { get; set; }

    public ApiServerHost(
        IModelRegistry registry,
        IInferenceService inferenceService,
        IOptions<OnnxStudioOptions> options,
        ILogger<ApiServerHost>? logger = null)
    {
        _registry = registry;
        _inferenceService = inferenceService;
        _options = options.Value;
        RequestedPort = _options.Api.Port;
        _logger = logger;
    }

    /// <summary>
    /// Starts the API server. Uses port 0 (dynamic) when the configured port is already taken.
    /// </summary>
    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        if (_app != null)
        {
            return;
        }

        var builder = WebApplication.CreateSlimBuilder();

        // Bind Kestrel via configuration and DI (ConfigureWebHostBuilder has no UseUrls)
        builder.Configuration["urls"] = $"http://localhost:{RequestedPort}";
        builder.Services.Configure<Microsoft.AspNetCore.Server.Kestrel.Core.KestrelServerOptions>(kestrel =>
        {
            kestrel.Limits.MaxRequestBodySize = (long)_options.Api.MaxRequestSizeMB * 1024 * 1024;
        });

        builder.Services.AddCors(options =>
        {
            options.AddPolicy("onnxstudio", policy => policy
                .AllowAnyOrigin()
                .AllowAnyMethod()
                .AllowAnyHeader());
        });

        // Same singleton instances as the host application
        builder.Services.AddSingleton(_registry);
        builder.Services.AddSingleton(_inferenceService);

        var app = builder.Build();
        app.UseCors("onnxstudio");
        app.MapOnnxStudioEndpoints();

        await app.StartAsync(cancellationToken);
        _app = app;

        var server = app.Services.GetRequiredService<Microsoft.AspNetCore.Hosting.Server.IServer>();
        var address = server.Features.Get<Microsoft.AspNetCore.Hosting.Server.Features.IServerAddressesFeature>();
        Port = ParsePort(address?.Addresses.FirstOrDefault());
        _logger?.LogInformation("ONNX Studio API listening on http://localhost:{Port}", Port);
    }

    public async Task StopAsync()
    {
        if (_app == null)
        {
            return;
        }
        await _app.StopAsync();
        await _app.DisposeAsync();
        _app = null;
    }

    public async ValueTask DisposeAsync()
    {
        await StopAsync();
    }

    private static int ParsePort(string? address)
    {
        if (address == null)
        {
            return 0;
        }
        var colon = address.LastIndexOf(':');
        return colon >= 0 && int.TryParse(address[(colon + 1)..], out var port) ? port : 0;
    }
}
