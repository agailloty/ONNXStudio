using ONNXStudio.Api;
using System.Text.Json;
using ONNXStudio.Api.Payloads;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Services;
using Xunit;

namespace ONNXStudio.UI.Tests;

public class TensorInputTests
{
    [Fact]
    public void IntegerInputPreservesPrecision()
    {
        var result = TensorInputParser.Parse(new("x", DataType.Int64, [1]), "9007199254740993");
        Assert.True(result.IsSuccess);
        Assert.Equal(9007199254740993L, Assert.IsType<long[]>(result.Value!.Data)[0]);
    }

    [Theory]
    [InlineData(DataType.Int64, "1.5")]
    [InlineData(DataType.Uint8, "256")]
    [InlineData(DataType.Float32, "NaN")]
    [InlineData(DataType.Float32, "1e100")]
    [InlineData(DataType.Float64, "Infinity")]
    public void InvalidValuesAreRejected(DataType type, string text) =>
        Assert.True(TensorInputParser.Parse(new("x", type, [1]), text).IsFailure);

    [Fact]
    public void ScalarKeepsRankZero() => Assert.Empty(TensorInputParser.Parse(new("x", DataType.Float32, []), "1").Value!.Shape!);

    [Fact]
    public void DynamicBatchIsInferredWithoutChangingRank()
    {
        var result = TensorInputParser.Parse(new("x", DataType.Float32, [null, 2, 2]), "1,2,3,4,5,6,7,8");
        Assert.Equal(new long[] { 2, 2, 2 }, result.Value!.Shape);
    }

    [Fact]
    public void SeveralDynamicDimensionsRequireExplicitShape()
    {
        var schema = new TensorSchema("x", DataType.Float32, [null, null]);
        Assert.True(TensorInputParser.Parse(schema, "1,2,3,4").IsFailure);
        Assert.Equal(new long[] { 2, 2 }, TensorInputParser.Parse(schema, "1,2,3,4", "2,2").Value!.Shape);
        Assert.True(TensorInputParser.Parse(schema, "1,2,3,4", "2,3").IsFailure);
    }

    [Fact]
    public void DynamicTokenInputIsAVector()
    {
        var fields = new FormGenerationService().GenerateFields(UiTestSetup.Model(new TensorSchema("tokens", DataType.Int64, [1, null])));
        Assert.Equal(FormFieldKind.Vector, fields[0].Kind);
    }

    [Fact]
    public void ApiExamplesEscapeNamesAndRespectDynamicShapes()
    {
        var model = UiTestSetup.Model(new TensorSchema("input\"with\\quotes", DataType.Float32, [null, 2, 3]));
        using var json = JsonDocument.Parse(ApiExamples.Payload(model));
        var parsed = PayloadParser.Parse(json, model);
        Assert.True(parsed.IsSuccess);
        Assert.Equal(new long[] { 1, 2, 3 }, parsed.Value!.Values.Single().Shape);
        using var schema = JsonDocument.Parse(ApiExamples.Schema(model));
        Assert.Equal("object", schema.RootElement.GetProperty("type").GetString());
    }

    [Fact]
    public void ApiRejectsRaggedArraysEvenWhenTotalElementCountMatches()
    {
        var model = UiTestSetup.Model(new TensorSchema("x", DataType.Float32, [2, 2]));
        using var json = JsonDocument.Parse("{\"inputs\":{\"x\":[[1,2],[3],[4]]}}");
        Assert.True(PayloadParser.Parse(json, model).IsFailure);
    }

    [Fact]
    public void ApiPreservesIntegerPrecision()
    {
        var model = UiTestSetup.Model(new TensorSchema("x", DataType.Int64, [1]));
        using var json = JsonDocument.Parse("{\"inputs\":{\"x\":[9007199254740993]}}");
        var parsed = PayloadParser.Parse(json, model);
        Assert.Equal(9007199254740993L, Assert.IsType<long[]>(parsed.Value!["x"].Data)[0]);
    }

    [Fact]
    public void OversizedExampleFailsWithoutAllocatingTensor() =>
        Assert.Throws<InvalidOperationException>(() => ApiExamples.Payload(UiTestSetup.Model(new TensorSchema("x", DataType.Float32, [long.MaxValue]))));
}
