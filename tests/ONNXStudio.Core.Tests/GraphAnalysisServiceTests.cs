using ONNXStudio.Core.Services;
using Xunit;

namespace ONNXStudio.Core.Tests;

public class GraphAnalysisServiceTests
{
    private readonly IGraphAnalysisService _service = new GraphAnalysisService();

    private async Task<Models.OnnxModel> LoadAsync(string fixture)
    {
        var result = await TestSetup.CreateLoader().LoadAsync(TestSetup.FixturePath(fixture));
        Assert.True(result.IsSuccess, result.Error?.ToString());
        return result.Value!;
    }

    [Fact]
    public async Task GetStatistics_OnConvnet_ReturnsCountsAndParameters()
    {
        var model = await LoadAsync("convnet.onnx");

        var stats = _service.GetStatistics(model);

        Assert.Equal(5, stats.NodeCount);
        Assert.Equal(4, stats.EdgeCount);
        Assert.Equal(5, stats.InitializerCount);
        Assert.Equal(1, stats.InputCount);
        Assert.Equal(1, stats.OutputCount);
        Assert.Equal(5, stats.DistinctOpTypes);

        // conv_w(216) + conv_b(8) + fc_w(80) + fc_b(10) + reshape_shape(2) = 316
        Assert.Equal(316, stats.TotalParameters);
    }

    [Fact]
    public async Task Search_ByOpType_FindsNode()
    {
        var model = await LoadAsync("convnet.onnx");

        var results = _service.Search(model, "relu", category: null);

        var node = Assert.Single(results);
        Assert.Equal("Relu", node.OpType);
    }

    [Fact]
    public async Task Search_ByCategory_FiltersNodes()
    {
        var model = await LoadAsync("convnet.onnx");

        var activations = _service.Search(model, null, "Activation");

        Assert.Equal("Relu", Assert.Single(activations).OpType);
    }

    [Fact]
    public async Task GetCategories_ReturnsDistinctSortedCategories()
    {
        var model = await LoadAsync("convnet.onnx");

        var categories = _service.GetCategories(model);

        Assert.Equal("All", categories[0]);
        Assert.Contains("Conv", categories);
        Assert.Contains("Linear", categories);
        Assert.Equal(categories.Skip(1), categories.Skip(1).OrderBy(c => c, StringComparer.Ordinal));
    }

    [Fact]
    public async Task GetDependencies_ReturnsProducersAndConsumers()
    {
        var model = await LoadAsync("convnet.onnx");

        // node_1 is Relu: depends on Conv (node_0), consumed by GlobalAveragePool (node_2)
        var deps = _service.GetDependencies(model, "node_1");

        Assert.Equal("Conv", Assert.Single(deps.DependsOn).OpType);
        Assert.Equal("GlobalAveragePool", Assert.Single(deps.DependedBy).OpType);
    }

    [Theory]
    [InlineData("Conv", "Conv")]
    [InlineData("MaxPool", "Pool")]
    [InlineData("Relu", "Activation")]
    [InlineData("MatMul", "Linear")]
    [InlineData("BatchNormalization", "Normalization")]
    [InlineData("Scaler", "Pipeline")]
    [InlineData("Unknown", "Other")]
    public void GetCategory_MapsOperators(string opType, string expected)
    {
        Assert.Equal(expected, _service.GetCategory(opType));
    }
}
