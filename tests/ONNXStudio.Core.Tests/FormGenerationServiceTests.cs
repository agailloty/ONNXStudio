using Microsoft.Extensions.Options;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using Xunit;

namespace ONNXStudio.Core.Tests;

public class FormGenerationServiceTests
{
    private readonly IFormGenerationService _service = new FormGenerationService();

    private static OnnxModel CreateModel(params TensorSchema[] inputs) => new(
        "id", "test.onnx", 1, "p", "", 15, "1", "", 8,
        new ComputationGraph(Array.Empty<GraphNode>(), Array.Empty<GraphEdge>()),
        inputs, Array.Empty<TensorSchema>());

    [Fact]
    public void ScalarNumericInput_GeneratesNumberField()
    {
        var model = CreateModel(new TensorSchema("lr", DataType.Float32, new long?[] { 1 }));
        var fields = _service.GenerateFields(model);

        var field = Assert.Single(fields);
        Assert.Equal("lr", field.InputName);
        Assert.Equal(FormFieldKind.Number, field.Kind);
    }

    [Fact]
    public void MultiElementVectorInput_GeneratesVectorField()
    {
        var model = CreateModel(new TensorSchema("X", DataType.Float32, new long?[] { 1, 4 }));
        var fields = _service.GenerateFields(model);

        var field = Assert.Single(fields);
        Assert.Equal(FormFieldKind.Vector, field.Kind);
        Assert.Contains("4 comma-separated", field.Description);
    }

    [Fact]
    public void Rank4Tensor_GeneratesImageFieldWithExpectedDimensions()
    {
        var model = CreateModel(new TensorSchema("image", DataType.Float32, new long?[] { 1, 3, 224, 224 }));
        var fields = _service.GenerateFields(model);

        var field = Assert.Single(fields);
        Assert.Equal(FormFieldKind.Image, field.Kind);
        Assert.Equal(3, field.ExpectedChannels);
        Assert.Equal(224, field.ExpectedHeight);
        Assert.Equal(224, field.ExpectedWidth);
    }

    [Fact]
    public void StringInput_GeneratesTextField()
    {
        var model = CreateModel(new TensorSchema("text", DataType.String, new long?[] { 1 }));
        var fields = _service.GenerateFields(model);

        var field = Assert.Single(fields);
        Assert.Equal(FormFieldKind.Text, field.Kind);
    }

    [Fact]
    public void DynamicBatchVector_GeneratesVectorField()
    {
        var model = CreateModel(new TensorSchema("X", DataType.Float32, new long?[] { null, 8 }));
        var fields = _service.GenerateFields(model);

        Assert.Equal(FormFieldKind.Vector, Assert.Single(fields).Kind);
    }
}

public class InferenceSessionManagerTests : IDisposable
{
    private InferenceSessionManager CreateManager(int capacity)
        => new(Options.Create(new OnnxStudioOptions { SessionCacheSize = capacity }));

    private static OnnxModel CreateModel(string path)
        => new("id_" + path, path, 1, "p", "", 15, "1", "", 8,
            new ComputationGraph(Array.Empty<GraphNode>(), Array.Empty<GraphEdge>()),
            Array.Empty<TensorSchema>(), Array.Empty<TensorSchema>());

    [Fact]
    public async Task CapacityEvictsLeastRecentlyUsedSession()
    {
        var manager = CreateManager(1);
        var loader = TestSetup.CreateLoader();

        var modelA = (await loader.LoadAsync(TestSetup.FixturePath("add.onnx"))).Value!;
        var modelB = (await loader.LoadAsync(TestSetup.FixturePath("linreg.onnx"))).Value!;

        var sessionA = manager.GetSession(modelA);
        Assert.Equal(1, manager.CachedSessionCount);

        manager.GetSession(modelB);
        Assert.Equal(1, manager.CachedSessionCount);

        // Session A was evicted (disposed); asking for it again creates a new one
        var sessionA2 = manager.GetSession(modelA);
        Assert.NotSame(sessionA, sessionA2);
        manager.Dispose();
    }

    [Fact]
    public void Evict_RemovesSession()
    {
        var manager = CreateManager(5);
        var loader = TestSetup.CreateLoader();
        var model = loader.LoadAsync(TestSetup.FixturePath("add.onnx")).Result.Value!;

        manager.GetSession(model);
        Assert.Equal(1, manager.CachedSessionCount);

        manager.Evict(model.FilePath);
        Assert.Equal(0, manager.CachedSessionCount);
        manager.Dispose();
    }

    public void Dispose()
    {
        // Managers created per test dispose themselves in the test body
    }
}
