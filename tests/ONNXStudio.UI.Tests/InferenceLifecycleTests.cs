using System.Net;
using System.Net.Sockets;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using ONNXStudio.Api;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Services;
using Xunit;

namespace ONNXStudio.UI.Tests;

public class InferenceLifecycleTests
{
    [Fact]
    public async Task UnloadEvictsSessionButActiveLeaseRemainsUsable()
    {
        await using var services = UiTestSetup.Services();
        var registry = services.GetRequiredService<IModelRegistry>();
        var model = await UiTestSetup.Load(services);
        registry.Register(model);
        var manager = services.GetRequiredService<IInferenceSessionManager>();
        using var lease = manager.Acquire(model);
        registry.Unload(model.Id);
        Assert.Equal(0, manager.CachedSessionCount);
        using var a = Microsoft.ML.OnnxRuntime.OrtValue.CreateTensorValueFromMemory(new[] { 1f, 2f }, new long[] { 2 });
        using var b = Microsoft.ML.OnnxRuntime.OrtValue.CreateTensorValueFromMemory(new[] { 3f, 4f }, new long[] { 2 });
        using var options = new Microsoft.ML.OnnxRuntime.RunOptions();
        using var outputs = lease.Session.Run(options, new Dictionary<string, Microsoft.ML.OnnxRuntime.OrtValue> { ["a"] = a, ["b"] = b }, lease.Session.OutputNames);
        Assert.Equal(new[] { 4f, 6f }, outputs.First().GetTensorDataAsSpan<float>().ToArray());
    }

    [Fact]
    public async Task EvictionDoesNotInvalidateAnotherModelsActiveSession()
    {
        await using var services = UiTestSetup.Services();
        using var manager = new InferenceSessionManager(Options.Create(new OnnxStudioOptions { SessionCacheSize = 1 }));
        var modelA = await UiTestSetup.Load(services);
        var modelB = await UiTestSetup.Load(services, "linreg.onnx");
        using var a = manager.Acquire(modelA);
        using var b = manager.Acquire(modelB);
        Assert.Equal(1, manager.CachedSessionCount);
        Assert.Equal(2, a.Session.InputNames.Count);
        using var replacement = manager.Acquire(modelA);
        Assert.NotSame(a.Session, replacement.Session);
    }

    [Fact]
    public async Task ServerStartIsSerializedAndRecoversAfterPortConflict()
    {
        await using var services = UiTestSetup.Services();
        var host = services.GetRequiredService<ApiServerHost>();
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        host.RequestedPort = ((IPEndPoint)listener.LocalEndpoint).Port;
        await Assert.ThrowsAnyAsync<Exception>(() => host.StartAsync(TestContext.Current.CancellationToken));
        Assert.False(host.IsRunning);
        host.RequestedPort = 0;
        await Task.WhenAll(host.StartAsync(TestContext.Current.CancellationToken), host.StartAsync(TestContext.Current.CancellationToken));
        Assert.True(host.Port > 0);
        await host.StopAsync();
        Assert.False(host.IsRunning);
    }

    [Theory]
    [InlineData(1, true)]
    [InlineData(7, false)]
    [InlineData(8, false)]
    [InlineData(9, false)]
    [InlineData(10, false)]
    public async Task ScalarAndTypedTensorsRoundTripThroughOnnx(int onnxType, bool scalar)
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".onnx");
        try
        {
            File.WriteAllBytes(path, IdentityModel(onnxType, scalar));
            await using var services = UiTestSetup.Services();
            var loaded = await services.GetRequiredService<IModelLoader>().LoadAsync(path, TestContext.Current.CancellationToken);
            Assert.True(loaded.IsSuccess, loaded.Error?.Message);
            Array data = onnxType switch { 7 => new[] { 9007199254740993L }, 8 => new[] { "Bonjour ONNX" }, 9 => new[] { true }, _ => new[] { 2.5f } };
            var result = await services.GetRequiredService<IInferenceService>().RunAsync(loaded.Value!,
                new Dictionary<string, InferenceInputValue> { ["x"] = new(data, scalar ? [] : [1]) }, TestContext.Current.CancellationToken);
            Assert.True(result.IsSuccess, result.Error?.Message);
            var output = Assert.Single(result.Value!.Outputs);
            Assert.Equal(data.GetValue(0), output.Data.GetValue(0));
            Assert.Equal(scalar ? 0 : 1, output.Shape.Count);
            Assert.Equal(data.GetValue(0), ONNXStudio.Api.Endpoints.OnnxStudioEndpoints.PredictResponse.From(loaded.Value!, result.Value!).Outputs["y"]);
        }
        finally { File.Delete(path); }
    }

    // Minimal real ModelProto: Identity(x)->y with a typed scalar or [1] tensor.
    private static byte[] IdentityModel(int type, bool scalar)
    {
        byte[] ValueInfo(string name)
        {
            var shape = scalar ? Array.Empty<byte>() : Message(1, Number(1, 1));
            var tensorType = Join(Number(1, type), Message(2, shape));
            return Join(Text(1, name), Message(2, Message(1, tensorType)));
        }
        var node = Join(Text(1, "x"), Text(2, "y"), Text(3, "identity"), Text(4, "Identity"));
        var graph = Join(Message(1, node), Text(2, "identity_graph"), Message(11, ValueInfo("x")), Message(12, ValueInfo("y")));
        return Join(Number(1, 8), Message(7, graph), Message(8, Number(2, 18)));
    }
    private static byte[] Text(int field, string value) => Message(field, Encoding.UTF8.GetBytes(value));
    private static byte[] Message(int field, byte[] value) => Join(Varint(field * 8 + 2), Varint(value.Length), value);
    private static byte[] Number(int field, int value) => Join(Varint(field * 8), Varint(value));
    private static byte[] Join(params byte[][] arrays) => arrays.SelectMany(a => a).ToArray();
    private static byte[] Varint(int value)
    {
        var result = new List<byte>();
        while (value >= 128) { result.Add((byte)(value | 128)); value >>= 7; }
        result.Add((byte)value);
        return result.ToArray();
    }
}
