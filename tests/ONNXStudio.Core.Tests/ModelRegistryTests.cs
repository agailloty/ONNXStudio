using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using Xunit;

namespace ONNXStudio.Core.Tests;

public class ModelRegistryTests
{
    private static OnnxModel CreateModel(string path, string id) => new(
        id, path, 100, "test", "", 15, "1", "", 8,
        new ComputationGraph(Array.Empty<GraphNode>(), Array.Empty<GraphEdge>()),
        Array.Empty<TensorSchema>(), Array.Empty<TensorSchema>());

    [Fact]
    public void Register_AddsModelAndRaisesEvent()
    {
        var registry = new ModelRegistry();
        OnnxModel? added = null;
        registry.ModelAdded += (_, m) => added = m;

        var model = CreateModel("a.onnx", "1");
        registry.Register(model);

        Assert.Same(model, added);
        Assert.Single(registry.Models);
        Assert.Same(model, registry.GetById("1"));
    }

    [Fact]
    public void Register_SameFile_ReplacesPreviousEntry()
    {
        var registry = new ModelRegistry();
        var first = registry.Register(CreateModel("a.onnx", "1"));
        var second = registry.Register(CreateModel("a.onnx", "2"));

        Assert.Single(registry.Models);
        Assert.Null(registry.GetById("1"));
        Assert.Same(second, registry.GetById("2"));
        Assert.True(registry.IsLoaded("a.onnx"));
    }

    [Fact]
    public void Unload_RemovesModelAndRaisesEvent()
    {
        var registry = new ModelRegistry();
        var model = registry.Register(CreateModel("a.onnx", "1"));
        OnnxModel? removed = null;
        registry.ModelRemoved += (_, m) => removed = m;

        var success = registry.Unload("1");

        Assert.True(success);
        Assert.Same(model, removed);
        Assert.Empty(registry.Models);
        Assert.False(registry.IsLoaded("a.onnx"));
    }

    [Fact]
    public void Unload_UnknownId_ReturnsFalse()
    {
        var registry = new ModelRegistry();
        Assert.False(registry.Unload("nope"));
    }
}
