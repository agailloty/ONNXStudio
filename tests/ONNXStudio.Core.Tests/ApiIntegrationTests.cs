using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using ONNXStudio.Api;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using Xunit;

namespace ONNXStudio.Core.Tests;

/// <summary>
/// Full-stack integration tests: real Kestrel server + real ONNX inference
/// over real HTTP (US-005, US-006).
/// </summary>
public class ApiIntegrationTests : IAsyncLifetime
{
    private readonly ModelRegistry _registry = new();
    private ApiServerHost? _host;
    private HttpClient _client = new();
    private OnnxModel? _addModel;
    private OnnxModel? _linregModel;

    public async Task InitializeAsync()
    {
        var loader = TestSetup.CreateLoader();
        _addModel = (await loader.LoadAsync(TestSetup.FixturePath("add.onnx"))).Value!;
        _linregModel = (await loader.LoadAsync(TestSetup.FixturePath("linreg.onnx"))).Value!;
        _registry.Register(_addModel);
        _registry.Register(_linregModel);

        var options = Microsoft.Extensions.Options.Options.Create(new OnnxStudioOptions
        {
            Api = new OnnxStudioApiOptions { Port = GetFreePort() }
        });
        var inference = new InferenceService(
            new InferenceSessionManager(options), options, NullLogger<InferenceService>.Instance);
        _host = new ApiServerHost(_registry, inference, options, NullLogger<ApiServerHost>.Instance);
        await _host.StartAsync();

        _client = new HttpClient { BaseAddress = new Uri($"http://localhost:{_host.Port}") };
    }

    private static int GetFreePort()
    {
        var listener = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
        listener.Start();
        int port = ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    public async Task DisposeAsync()
    {
        _client.Dispose();
        if (_host != null)
        {
            await _host.DisposeAsync();
        }
    }

    [Fact]
    public async Task Health_ReturnsOk()
    {
        var response = await _client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal("ok", body.GetProperty("status").GetString());
    }

    [Fact]
    public async Task GetModels_ListsRegisteredModels()
    {
        var response = await _client.GetAsync("/models");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal(2, body.GetArrayLength());
        var names = body.EnumerateArray()
            .Select(item => item.GetProperty("name").GetString())
            .ToList();
        Assert.Contains("add", names);
        Assert.Contains("linreg", names);
    }

    [Fact]
    public async Task GetModelDetails_ReturnsMetadata()
    {
        var response = await _client.GetAsync($"/models/{_linregModel!.Id}");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal("linreg", body.GetProperty("name").GetString());
        Assert.Equal(16, body.GetProperty("opsetVersion").GetInt64());
        Assert.Equal(1, body.GetProperty("inputs").GetArrayLength());
    }

    [Fact]
    public async Task GetUnknownModel_Returns404()
    {
        var response = await _client.GetAsync("/models/nope");
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task Predict_OnAddModel_ComputesSum()
    {
        var payload = new
        {
            inputs = new Dictionary<string, object>
            {
                ["a"] = new[] { 1, 2 },
                ["b"] = new[] { 3, 4 }
            }
        };

        var response = await _client.PostAsJsonAsync($"/models/{_addModel!.Id}/predict", payload);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        var y = body.GetProperty("outputs").GetProperty("y");
        Assert.Equal(4, y[0].GetDouble());
        Assert.Equal(6, y[1].GetDouble());
        Assert.True(body.GetProperty("executionTimeMs").GetInt64() >= 0);
    }

    [Fact]
    public async Task Predict_OnLinreg_AcceptsNestedArray()
    {
        var payload = new
        {
            inputs = new Dictionary<string, object>
            {
                ["X"] = new[] { new[] { 1.0, 1.0, 1.0, 1.0 } }
            }
        };

        var response = await _client.PostAsJsonAsync($"/models/{_linregModel!.Id}/predict", payload);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        var y = body.GetProperty("outputs").GetProperty("y");
        Assert.Equal(10.5, y.GetDouble(), precision: 4);
    }

    [Fact]
    public async Task Predict_MissingInput_Returns400()
    {
        var payload = new { inputs = new Dictionary<string, object> { ["a"] = new[] { 1, 2 } } };

        var response = await _client.PostAsJsonAsync($"/models/{_addModel!.Id}/predict", payload);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Contains("b", body.GetProperty("error").GetString());
    }

    [Fact]
    public async Task Predict_UnknownInput_Returns400()
    {
        var payload = new { inputs = new Dictionary<string, object> { ["z"] = new[] { 1, 2 } } };

        var response = await _client.PostAsJsonAsync($"/models/{_addModel!.Id}/predict", payload);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Contains("Unknown input", body.GetProperty("error").GetString());
    }

    [Fact]
    public async Task Predict_InvalidJson_Returns400()
    {
        var response = await _client.PostAsync($"/models/{_addModel!.Id}/predict",
            new StringContent("not json", System.Text.Encoding.UTF8, "application/json"));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Predict_UnknownModel_Returns404()
    {
        var response = await _client.PostAsJsonAsync("/models/unknown/predict",
            new { inputs = new Dictionary<string, object>() });
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task Schema_ReturnsInputOutputShapes()
    {
        var response = await _client.GetAsync($"/models/{_addModel!.Id}/schema");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal(2, body.GetProperty("inputs").GetArrayLength());
        Assert.Equal(1, body.GetProperty("outputs").GetArrayLength());
        Assert.NotNull(body.GetProperty("requestExample").GetString());
    }
}
