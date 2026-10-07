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
    private readonly SemaphoreSlim _lifecycle = new(1);
    public event Action? StateChanged;

    public int Port { get; private set; }
    public bool IsRunning => _app != null;

    /// <summary>
    /// Port used on the next start (configurable from the API config screen).
    /// Defaults to the configured options port when not overridden.
    /// </summary>
    private int _requestedPort;
    public int RequestedPort
    {
        get => _requestedPort;
        set
        {
            if (value is < 0 or > 65535) throw new ArgumentOutOfRangeException(nameof(value), "Port must be between 0 and 65535.");
            if (_requestedPort == value) return;
            _requestedPort = value;
            StateChanged?.Invoke();
        }
    }

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
        _registry.ModelRemoved += OnModelRemoved;
    }

    /// <summary>
    /// Starts the API server. Port 0 selects an available loopback port.
    /// </summary>
    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        await _lifecycle.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_app != null)
            {
                return;
            }

            var builder = WebApplication.CreateSlimBuilder();

            // Bind Kestrel via configuration and DI (ConfigureWebHostBuilder has no UseUrls)
            builder.Configuration["urls"] = RequestedPort == 0 ? "http://127.0.0.1:0" : $"http://localhost:{RequestedPort}";
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

            try { await app.StartAsync(cancellationToken).ConfigureAwait(false); }
            catch { await app.DisposeAsync().ConfigureAwait(false); throw; }
            _app = app;

            var server = app.Services.GetRequiredService<Microsoft.AspNetCore.Hosting.Server.IServer>();
            var address = server.Features.Get<Microsoft.AspNetCore.Hosting.Server.Features.IServerAddressesFeature>();
            Port = ParsePort(address?.Addresses.FirstOrDefault());
            _logger?.LogInformation("ONNX Studio API listening on http://localhost:{Port}", Port);
        }
        finally { _lifecycle.Release(); }
        StateChanged?.Invoke();
    }

    public async Task StopAsync()
    {
        await _lifecycle.WaitAsync().ConfigureAwait(false);
        try
        {
            if (_app == null) return;
            var app = _app;
            try { await app.StopAsync().ConfigureAwait(false); }
            finally
            {
                await app.DisposeAsync().ConfigureAwait(false);
                _app = null;
                Port = 0;
            }
        }
        finally { _lifecycle.Release(); }
        StateChanged?.Invoke();
    }

    public async ValueTask DisposeAsync()
    {
        _registry.ModelRemoved -= OnModelRemoved;
        await StopAsync();
    }

    private async void OnModelRemoved(object? sender, ONNXStudio.Core.Models.OnnxModel model)
    {
        try
        {
            if (_registry.Models.Count == 0) await StopAsync().ConfigureAwait(false);
        }
        catch (Exception ex) { _logger?.LogError(ex, "Could not stop the API after unloading the last model"); }
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
