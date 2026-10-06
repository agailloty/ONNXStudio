using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Services;
using Xunit;

namespace ONNXStudio.Core.Tests;

public class InferenceServiceTests : IDisposable
{
    private readonly IInferenceSessionManager _sessionManager =
        new InferenceSessionManager(Options.Create(new OnnxStudioOptions()));

    private InferenceService CreateService(int maxConcurrent = 4)
        => new(_sessionManager, Options.Create(new OnnxStudioOptions { MaxConcurrentInferences = maxConcurrent }),
            NullLogger<InferenceService>.Instance);

    private async Task<Models.OnnxModel> LoadAsync(string fixture)
    {
        var loader = TestSetup.CreateLoader();
        var result = await loader.LoadAsync(TestSetup.FixturePath(fixture));
        Assert.True(result.IsSuccess, result.Error?.ToString());
        return result.Value!;
    }

    [Fact]
    public async Task RunAddModel_ComputesElementWiseSum()
    {
        var model = await LoadAsync("add.onnx");
        var service = CreateService();

        var result = await service.RunAsync(model, new Dictionary<string, InferenceInputValue>
        {
            ["a"] = InferenceInputValue.Scalars(1, 2),
            ["b"] = InferenceInputValue.Scalars(3, 4)
        });

        Assert.True(result.IsSuccess, result.Error?.ToString());
        var output = result.Value!.Outputs.Single(o => o.Name == "y");
        var values = Assert.IsType<float[]>(output.Data);
        Assert.Equal(new[] { 4f, 6f }, values);
    }

    [Fact]
    public async Task RunLinregModel_ComputesLinearRegression()
    {
        var model = await LoadAsync("linreg.onnx");
        var service = CreateService();

        // y = X.W + C = (1*1 + 1*2 + 1*3 + 1*4) + 0.5 = 10.5
        var result = await service.RunAsync(model, new Dictionary<string, InferenceInputValue>
        {
            ["X"] = InferenceInputValue.Array(new[] { 1.0, 1.0, 1.0, 1.0 }, new long[] { 1, 4 })
        });

        Assert.True(result.IsSuccess, result.Error?.ToString());
        var output = result.Value!.Outputs.Single(o => o.Name == "y");
        Assert.Equal(new long[] { 1, 1 }, output.Shape);
        Assert.Equal(10.5f, Assert.IsType<float[]>(output.Data)[0], precision: 4);
    }

    [Fact]
    public async Task RunConvnetModel_ReturnsClassifierOutput()
    {
        var model = await LoadAsync("convnet.onnx");
        var service = CreateService();

        var pixels = new float[1 * 3 * 32 * 32];
        var result = await service.RunAsync(model, new Dictionary<string, InferenceInputValue>
        {
            ["X"] = InferenceInputValue.Tensor(pixels, new long[] { 1, 3, 32, 32 })
        });

        Assert.True(result.IsSuccess, result.Error?.ToString());
        var output = result.Value!.Outputs.Single(o => o.Name == "Y");
        Assert.Equal(new long[] { 1, 10 }, output.Shape);
        Assert.Equal(10, output.Data.Length);
    }

    [Fact]
    public async Task RunDynamicModel_AcceptsDynamicBatch()
    {
        var model = await LoadAsync("dynamic.onnx");
        var service = CreateService();

        var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var result = await service.RunAsync(model, new Dictionary<string, InferenceInputValue>
        {
            ["X"] = InferenceInputValue.Array(data, new long[] { 2, 4 })
        });

        Assert.True(result.IsSuccess, result.Error?.ToString());
        var output = result.Value!.Outputs.Single(o => o.Name == "Y");
        Assert.Equal(new long[] { 2, 2 }, output.Shape);
    }

    [Fact]
    public async Task RunMissingInput_ReturnsMissingInputError()
    {
        var model = await LoadAsync("add.onnx");
        var service = CreateService();

        var result = await service.RunAsync(model, new Dictionary<string, InferenceInputValue>
        {
            ["a"] = InferenceInputValue.Scalars(1, 2)
        });

        Assert.True(result.IsFailure);
        Assert.Equal(InferenceErrorCode.MissingInput, result.Error!.Code);
        Assert.Contains("b", result.Error.Message);
    }

    [Fact]
    public async Task RunWrongShape_ReturnsShapeMismatch()
    {
        var model = await LoadAsync("linreg.onnx");
        var service = CreateService();

        var result = await service.RunAsync(model, new Dictionary<string, InferenceInputValue>
        {
            ["X"] = InferenceInputValue.Array(new[] { 1.0, 1.0, 1.0, 1.0 }, new long[] { 4 })
        });

        Assert.True(result.IsFailure);
        Assert.Equal(InferenceErrorCode.ShapeMismatch, result.Error!.Code);
        Assert.Contains("X", result.Error.Message);
    }

    [Fact]
    public async Task RunWrongElementCount_ReturnsInvalidInput()
    {
        var model = await LoadAsync("linreg.onnx");
        var service = CreateService();

        var result = await service.RunAsync(model, new Dictionary<string, InferenceInputValue>
        {
            ["X"] = InferenceInputValue.Array(new[] { 1.0, 2.0, 3.0 }, new long[] { 1, 4 })
        });

        Assert.True(result.IsFailure);
        Assert.Equal(InferenceErrorCode.InvalidInput, result.Error!.Code);
    }

    public void Dispose() => _sessionManager.Dispose();
}
