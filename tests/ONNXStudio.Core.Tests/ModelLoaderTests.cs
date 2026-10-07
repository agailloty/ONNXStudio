using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using Xunit;

namespace ONNXStudio.Core.Tests;

public class ModelLoaderTests
{
    [Fact]
    public async Task LoadAddModel_SucceedsWithMetadata()
    {
        var loader = TestSetup.CreateLoader();

        var result = await loader.LoadAsync(TestSetup.FixturePath("add.onnx"));

        Assert.True(result.IsSuccess, result.Error?.ToString());
        var model = result.Value!;
        Assert.Equal("add", model.Name);
        Assert.Equal("onnx-fixtures", model.ProducerName);
        Assert.Equal(15, model.OpsetVersion);
        Assert.Equal(8, model.IrVersion);
        Assert.Equal(2, model.Inputs.Count);
        Assert.Single(model.Outputs);
        Assert.Equal("y", model.Outputs[0].Name);
        Assert.Equal(DataType.Float32, model.Outputs[0].Type);
        Assert.Single(model.Outputs[0].Shape);
        Assert.Equal(2, model.Outputs[0].Shape[0]);
    }

    [Fact]
    public async Task LoadScikitLearnConversion_ReadsTheMetadataStoredByTheConverter()
    {
        // sklearn_pipeline.onnx: StandardScaler -> LogisticRegression converted by ONNX Studio's Python worker.
        var loader = TestSetup.CreateLoader();

        var result = await loader.LoadAsync(TestSetup.FixturePath("sklearn_pipeline.onnx"));

        Assert.True(result.IsSuccess, result.Error?.ToString());
        var metadata = result.Value!.Metadata;
        Assert.Equal("scikit-learn", metadata["onnxstudio.source"]);
        Assert.Equal("Pipeline", metadata["onnxstudio.sklearn.class"]);
        var info = ONNXStudio.Core.Python.PythonModelInfo.TryParse(metadata[ONNXStudio.Core.Python.PythonModelInfo.OnnxMetadataKey]);
        var root = info!.Components!;
        Assert.Equal("Pipeline", root.Kind);
        Assert.Equal(new[] { "scaler", "clf" }, root.Children.Select(c => c.Name));
        var coefficients = root.Children[1].Fitted.Single(f => f.Name == "coef_");
        Assert.Equal(new[] { 1, 4 }, coefficients.Shape);
        Assert.Equal(4, coefficients.Values.Count);
        Assert.Contains(root.Children[1].Parameters, p => p.Key == "C");
    }

    [Fact]
    public void ModelsWithoutMetadataAndInvalidJsonAreHandled()
    {
        Assert.Null(ONNXStudio.Core.Python.PythonModelInfo.TryParse("not json"));
        Assert.Null(ONNXStudio.Core.Python.PythonModelInfo.TryParse("[1, 2]"));
    }

    [Fact]
    public async Task LoadLinregModel_ExtractsGraphAndAttributes()
    {
        var loader = TestSetup.CreateLoader();

        var result = await loader.LoadAsync(TestSetup.FixturePath("linreg.onnx"));

        Assert.True(result.IsSuccess, result.Error?.ToString());
        var model = result.Value!;
        Assert.Single(model.Graph.Nodes);
        var node = model.Graph.Nodes[0];
        Assert.Equal("Gemm", node.OpType);
        Assert.Contains("alpha", node.Attributes);
        Assert.Equal(1.0f, Assert.IsType<float>(node.Attributes["alpha"]));
        Assert.Equal(1.0f, Assert.IsType<float>(node.Attributes["beta"]));

        var input = model.Inputs.Single(i => i.Name == "X");
        Assert.Equal("float32[1, 4]", input.ToDisplayString());
        Assert.False(input.HasDynamicDimension);
    }

    [Fact]
    public async Task LoadConvnetModel_BuildsTensorFlowEdges()
    {
        var loader = TestSetup.CreateLoader();

        var result = await loader.LoadAsync(TestSetup.FixturePath("convnet.onnx"));

        Assert.True(result.IsSuccess, result.Error?.ToString());
        var model = result.Value!;
        Assert.Equal(5, model.Graph.Nodes.Count);
        Assert.Equal("Conv", model.Graph.Nodes[0].OpType);

        var conv = model.Graph.Nodes[0];
        Assert.Equal(new List<long> { 3, 3 }, Assert.IsType<List<long>>(conv.Attributes["kernel_shape"]));

        // 5 chained nodes => 4 tensor-flow edges, in consumer order
        Assert.Equal(4, model.Graph.Edges.Count);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal($"node_{i}", model.Graph.Edges[i].FromNodeId);
            Assert.Equal($"node_{i + 1}", model.Graph.Edges[i].ToNodeId);
        }

        var input = model.Inputs[0];
        Assert.Equal("float32[1, 3, 32, 32]", input.ToDisplayString());
        Assert.Equal("float32[1, 10]", model.Outputs[0].ToDisplayString());
    }

    [Fact]
    public async Task LoadDynamicModel_MarksDynamicDimensions()
    {
        var loader = TestSetup.CreateLoader();

        var result = await loader.LoadAsync(TestSetup.FixturePath("dynamic.onnx"));

        Assert.True(result.IsSuccess, result.Error?.ToString());
        var input = result.Value!.Inputs[0];
        Assert.True(input.HasDynamicDimension);
        Assert.Null(input.Shape[0]);
        Assert.Equal(4, input.Shape[1]);
        Assert.Equal("float32[?, 4]", input.ToDisplayString());
    }

    [Fact]
    public async Task LoadNonExistentFile_ReturnsFileNotFound()
    {
        var loader = TestSetup.CreateLoader();

        var result = await loader.LoadAsync(TestSetup.FixturePath("does-not-exist.onnx"));

        Assert.True(result.IsFailure);
        Assert.Equal(ModelLoadErrorCode.FileNotFound, result.Error!.Code);
    }

    [Fact]
    public async Task LoadWrongExtension_ReturnsInvalidFileExtension()
    {
        var path = Path.ChangeExtension(TestSetup.FixturePath("add.onnx"), ".pdf");
        File.Copy(TestSetup.FixturePath("add.onnx"), path, true);

        try
        {
            var loader = TestSetup.CreateLoader();
            var result = await loader.LoadAsync(path);

            Assert.True(result.IsFailure);
            Assert.Equal(ModelLoadErrorCode.InvalidFileExtension, result.Error!.Code);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task LoadTooLargeFile_ReturnsFileTooLarge()
    {
        var options = new OnnxStudioOptions { MaxModelSizeMB = 0 };
        var loader = TestSetup.CreateLoader(options);

        var result = await loader.LoadAsync(TestSetup.FixturePath("add.onnx"));

        Assert.True(result.IsFailure);
        Assert.Equal(ModelLoadErrorCode.FileTooLarge, result.Error!.Code);
    }

    [Fact]
    public async Task LoadCorruptedFile_ReturnsInvalidFileFormat()
    {
        var path = Path.Combine(Path.GetTempPath(), "corrupted.onnx");
        File.WriteAllBytes(path, new byte[] { 0x07, 0xDE, 0xAD, 0xBE, 0xEF });

        try
        {
            var loader = TestSetup.CreateLoader();
            var result = await loader.LoadAsync(path);

            Assert.True(result.IsFailure);
            Assert.Equal(ModelLoadErrorCode.InvalidFileFormat, result.Error!.Code);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task LoadUnsupportedOpset_ReturnsUnsupportedOpsetVersionWithSuggestion()
    {
        var loader = TestSetup.CreateLoader();

        var result = await loader.LoadAsync(TestSetup.FixturePath("opset19.onnx"));

        Assert.True(result.IsFailure);
        Assert.Equal(ModelLoadErrorCode.UnsupportedOpsetVersion, result.Error!.Code);
        Assert.Contains("opset", result.Error.Message);
        Assert.Contains("18", result.Error.Message); // actionable suggestion
    }

    [Fact]
    public async Task LoadModelWithoutInputs_ReturnsInvalidModel()
    {
        var loader = TestSetup.CreateLoader();

        var result = await loader.LoadAsync(TestSetup.FixturePath("empty.onnx"));

        Assert.True(result.IsFailure);
        Assert.Equal(ModelLoadErrorCode.InvalidModel, result.Error!.Code);
    }
}
