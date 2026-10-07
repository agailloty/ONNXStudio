using System.Text.Json.Nodes;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Python;
using ONNXStudio.Core.Services;
using Xunit;

namespace ONNXStudio.Core.Tests;

public class SklearnModelTests
{
    private static SklearnModel Model(PythonModelInfo info) =>
        SklearnModel.Create(new PythonModel("source", "pipeline.joblib", 100), info);

    [Fact]
    public void ParallelTransformersMergeBeforeTheFinalEstimatorAndLinkToStructure()
    {
        var model = Model(new PythonModelInfo
        {
            FeatureCount = 2,
            Components = new()
            {
                Name = "pipeline", ClassName = "Pipeline", Kind = "Pipeline",
                Children =
                [
                    new()
                    {
                        Name = "features", ClassName = "ColumnTransformer", Kind = "ColumnTransformer",
                        Children =
                        [
                            new() { Name = "numeric_", ClassName = "StandardScaler", Kind = "Transformer" },
                            new() { Name = "category_", ClassName = "OneHotEncoder", Kind = "Transformer" }
                        ]
                    },
                    new() { Name = "predict", ClassName = "LogisticRegression", Kind = "Classifier" }
                ]
            }
        });
        var analysis = new GraphAnalysisService();
        Assert.Equal(4, model.Graph.Nodes.Count);
        Assert.Equal(3, model.Graph.Edges.Count);
        var merge = model.Graph.Nodes.Single(n => n.OpType == "ColumnTransformer");
        var dependencies = analysis.GetDependencies(model, merge.Id);
        Assert.Equal(new[] { "StandardScaler", "OneHotEncoder" }, dependencies.DependsOn.Select(n => n.OpType));
        Assert.Equal("LogisticRegression", Assert.Single(dependencies.DependedBy).OpType);
        Assert.Equal(2, analysis.Search(model, null, "Transformer").Count);
        var featureTree = Assert.Single(model.Structure).Children[1].Children[0];
        Assert.Equal(merge.Id, featureTree.NodeId);
        Assert.All(featureTree.Children, c => Assert.Contains(model.Graph.Nodes, n => n.Id == c.NodeId));
    }

    [Fact]
    public async Task IntegerInputsAndTypedPredictionsSurviveThePythonAdapter()
    {
        var service = new PredictionService();
        var backend = new SklearnInferenceBackend(service);
        var model = Model(new PythonModelInfo { FeatureCount = 2, Methods = ["predict"] });
        var result = await backend.RunAsync(model, new Dictionary<string, InferenceInputValue>
        {
            ["X"] = InferenceInputValue.Integers([1, 2, 3, 4], [2, 2])
        });
        Assert.True(result.IsSuccess, result.Error?.Message);
        Assert.Equal(new[] { "1", "2" }, service.Request!.Rows[0]);
        Assert.Equal(new[] { "3", "4" }, service.Request.Rows[1]);
        var output = Assert.Single(result.Value!.Outputs);
        Assert.Equal(DataType.Int64, output.Type);
        Assert.Equal(new long[] { 2 }, output.Shape);
        Assert.Equal(new long[] { 7, 9 }, Assert.IsType<long[]>(output.Data));
    }

    [Fact]
    public async Task ColumnsWithDifferentRowCountsAreRejectedWithoutDroppingRows()
    {
        var service = new PredictionService();
        var model = Model(new PythonModelInfo
        {
            FeatureNames = ["a", "b"],
            Components = new() { Name = "features", ClassName = "ColumnTransformer", Kind = "ColumnTransformer" }
        });
        var result = await new SklearnInferenceBackend(service).RunAsync(model, new Dictionary<string, InferenceInputValue>
        {
            ["a"] = InferenceInputValue.Array([1, 2], [2, 1]),
            ["b"] = InferenceInputValue.Array([3], [1, 1])
        });
        Assert.True(result.IsFailure);
        Assert.Contains("same number of rows", result.Error!.Message);
        Assert.Null(service.Request);
    }

    private sealed class PredictionService : IPythonModelService
    {
        public PythonPredictionRequest? Request { get; private set; }

        public Task<Result<PythonPredictionResult, PythonError>> PredictAsync(PythonPredictionRequest request, CancellationToken cancellationToken = default)
        {
            Request = request;
            return Task.FromResult(Result<PythonPredictionResult, PythonError>.Success(new PythonPredictionResult(
                [new("predict", "int64", [2], false, ["7", "9"], JsonNode.Parse("[7,9]"))], [], 2, 1, [])));
        }

        public Task<Result<PythonModelInfo, PythonError>> InspectAsync(string path, bool trustConfirmed, CancellationToken cancellationToken = default)
            => throw new NotSupportedException();

        public Task<Result<PythonConversionResult, PythonError>> ConvertAsync(PythonConversionRequest request, CancellationToken cancellationToken = default)
            => throw new NotSupportedException();
    }
}
