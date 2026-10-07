using System.Globalization;
using System.Text.Json.Nodes;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;

namespace ONNXStudio.Core.Python;

/// <summary>Runs <see cref="SklearnModel"/>s in the Python worker: tensors are sent as table rows and every inference method of the model comes back as an output tensor.</summary>
public sealed class SklearnInferenceBackend : IInferenceBackend
{
    private readonly IPythonModelService _python;

    public SklearnInferenceBackend(IPythonModelService python) => _python = python;

    public bool CanRun(IModel model) => model is SklearnModel;

    public async Task<Result<InferenceResult, InferenceError>> RunAsync(
        IModel model,
        IReadOnlyDictionary<string, InferenceInputValue> inputs,
        CancellationToken cancellationToken = default)
    {
        var sklearn = (SklearnModel)model;
        if (sklearn.InputLayout == SklearnInputLayout.Columns &&
            sklearn.Inputs.Select(i => inputs[i.Name].Data.Length).Distinct().Count() != 1)
            return Result<InferenceResult, InferenceError>.Failure(new InferenceError(
                InferenceErrorCode.InvalidInput, "All input columns must contain the same number of rows."));
        var (columns, rows) = ToRows(sklearn, inputs);

        // The model was loaded after the user confirmed that the file is trusted.
        var response = await _python.PredictAsync(
            new PythonPredictionRequest(sklearn.FilePath, true, "all", columns, rows), cancellationToken).ConfigureAwait(false);
        if (response.IsFailure)
        {
            var error = response.Error!;
            return Result<InferenceResult, InferenceError>.Failure(new InferenceError(
                error.Code == PythonErrorCode.InvalidInput ? InferenceErrorCode.InvalidInput : InferenceErrorCode.InferenceFailed,
                error.Message, error.TechnicalDetails));
        }

        var prediction = response.Value!;
        return Result<InferenceResult, InferenceError>.Success(new InferenceResult(
            sklearn.Id, prediction.ElapsedMs, prediction.Outputs.Select(ToTensor).ToArray()));
    }

    private static (IReadOnlyList<string>? Columns, IReadOnlyList<IReadOnlyList<string>> Rows) ToRows(
        SklearnModel model, IReadOnlyDictionary<string, InferenceInputValue> inputs)
    {
        switch (model.InputLayout)
        {
            case SklearnInputLayout.Text:
                return (null, inputs[model.Inputs[0].Name].Data.Cast<object?>().Select(v => (IReadOnlyList<string>)[Cell(v)]).ToArray());

            case SklearnInputLayout.Columns:
            {
                var columns = model.Inputs.Select(i => i.Name).ToArray();
                var cells = columns.Select(c => inputs[c].Data.Cast<object?>().Select(Cell).ToArray()).ToArray();
                var count = cells.Min(c => c.Length);
                return (columns, Enumerable.Range(0, count).Select(r => (IReadOnlyList<string>)cells.Select(c => c[r]).ToArray()).ToArray());
            }

            default:
            {
                var value = inputs[model.Inputs[0].Name];
                var data = value.Data.Cast<object?>().Select(Cell).ToArray();
                var width = (int?)model.Inputs[0].Shape.ElementAtOrDefault(1) ?? (value.Shape is { Length: > 1 } ? (int)value.Shape[1] : data.Length);
                var rows = Enumerable.Range(0, Math.Max(1, data.Length / Math.Max(1, width)))
                    .Select(r => (IReadOnlyList<string>)data.Skip(r * width).Take(width).ToArray()).ToArray();
                return (model.Info.FeatureNames, rows);
            }
        }
    }

    private static string Cell(object? value) => value switch
    {
        null => string.Empty,
        double number => number.ToString("R", CultureInfo.InvariantCulture),
        float number => number.ToString("R", CultureInfo.InvariantCulture),
        IFormattable number => number.ToString(null, CultureInfo.InvariantCulture),
        _ => value.ToString() ?? string.Empty
    };

    private static TensorOutput ToTensor(PythonPredictionOutput output)
    {
        var shape = output.Shape.Select(d => (long)d).ToArray();
        var leaves = new List<JsonNode?>();
        Flatten(output.Raw, leaves);

        if (output.DType.StartsWith("int", StringComparison.Ordinal) || output.DType.StartsWith("uint", StringComparison.Ordinal))
            return new TensorOutput(output.Name, DataType.Int64, shape, leaves.Select(n => n is JsonValue v && v.TryGetValue<long>(out var l) ? l : 0L).ToArray());
        if (output.DType.StartsWith("float", StringComparison.Ordinal))
            return new TensorOutput(output.Name, DataType.Float64, shape, leaves.Select(n => n is JsonValue v && v.TryGetValue<double>(out var d) ? d : double.NaN).ToArray());
        if (output.DType == "bool")
            return new TensorOutput(output.Name, DataType.Bool, shape, leaves.Select(n => n is JsonValue v && v.TryGetValue<bool>(out var b) && b).ToArray());
        return new TensorOutput(output.Name, DataType.String, shape, leaves.Select(n => PythonPredictionOutput.Format(n)).ToArray());
    }

    private static void Flatten(JsonNode? node, List<JsonNode?> leaves)
    {
        if (node is JsonArray array)
        {
            foreach (var item in array) Flatten(item, leaves);
        }
        else
        {
            leaves.Add(node);
        }
    }
}
