using System.Text.Json;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;

namespace ONNXStudio.Api.Payloads;

/// <summary>
/// Parses the JSON inference request payload into validated
/// <see cref="InferenceInputValue"/> tensors (US-005).
///
/// Supported payload format:
/// {
///   "inputs": {
///     "inputName": 3.14,          // scalar
///     "other":   [1, 2, 3],      // flat array (rank 1)
///     "matrix":  [[1,2],[3,4]]   // nested arrays (rank 2+)
///   }
/// }
/// </summary>
public static class PayloadParser
{
    public static Result<IReadOnlyDictionary<string, InferenceInputValue>, string> Parse(
        JsonDocument document,
        OnnxModel model)
    {
        var root = document.RootElement;
        if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("inputs", out var inputsEl)
            || inputsEl.ValueKind != JsonValueKind.Object)
        {
            return Result<IReadOnlyDictionary<string, InferenceInputValue>, string>.Failure(
                "The request body must be a JSON object with an 'inputs' object.");
        }

        var schemaByName = model.Inputs.ToDictionary(i => i.Name, StringComparer.Ordinal);
        var values = new Dictionary<string, InferenceInputValue>();

        foreach (var property in inputsEl.EnumerateObject())
        {
            if (!schemaByName.TryGetValue(property.Name, out var schema))
            {
                return Result<IReadOnlyDictionary<string, InferenceInputValue>, string>.Failure(
                    $"Unknown input '{property.Name}'. Model inputs: {string.Join(", ", schemaByName.Keys)}.");
            }

            var parsed = ParseValue(property.Value, schema);
            if (parsed.IsFailure)
            {
                return Result<IReadOnlyDictionary<string, InferenceInputValue>, string>.Failure(
                    $"Invalid value for input '{property.Name}': {parsed.Error}");
            }
            values[schema.Name] = parsed.Value!;
        }

        return Result<IReadOnlyDictionary<string, InferenceInputValue>, string>.Success(values);
    }

    private static Result<InferenceInputValue, string> ParseValue(JsonElement element, TensorSchema schema)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Number:
                return Result<InferenceInputValue, string>.Success(
                    InferenceInputValue.Scalar(element.GetDouble()));

            case JsonValueKind.Array:
                return ParseArray(element, schema);

            default:
                return Result<InferenceInputValue, string>.Failure("expected a number or an array of numbers");
        }
    }

    private static Result<InferenceInputValue, string> ParseArray(JsonElement array, TensorSchema schema)
    {
        // Nested arrays => infer the shape recursively
        if (array.EnumerateArray().FirstOrDefault().ValueKind == JsonValueKind.Array)
        {
            var (flat, shape) = Flatten(array);
            if (flat is null)
            {
                return Result<InferenceInputValue, string>.Failure("nested arrays must only contain numbers");
            }
            return Result<InferenceInputValue, string>.Success(InferenceInputValue.Array(flat, shape));
        }

        // Flat array of numbers
        var data = new List<double>();
        foreach (var item in array.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Number)
            {
                return Result<InferenceInputValue, string>.Failure("arrays must only contain numbers");
            }
            data.Add(item.GetDouble());
        }

        // Rank 1 tensors use [length]; higher ranks reuse the schema shape
        if (schema.Shape.Count == 1)
        {
            return Result<InferenceInputValue, string>.Success(
                InferenceInputValue.Array(data.ToArray(), new[] { (long)data.Count }));
        }

        var dims = new long[schema.Shape.Count];
        for (int i = 0; i < schema.Shape.Count; i++)
        {
            if (!schema.Shape[i].HasValue)
            {
                return Result<InferenceInputValue, string>.Failure(
                    $"flat arrays are not supported for dynamic input '{schema.Name}' with rank {schema.Shape.Count}; send a nested array");
            }
            dims[i] = schema.Shape[i]!.Value;
        }

        return Result<InferenceInputValue, string>.Success(InferenceInputValue.Array(data.ToArray(), dims));
    }

    private static (double[]? Flat, long[] Shape) Flatten(JsonElement element)
    {
        // Phase 1: infer the shape by walking the leftmost branch
        var dims = new List<long>();
        var probe = element;
        while (probe.ValueKind == JsonValueKind.Array)
        {
            long length = probe.GetArrayLength();
            if (length == 0)
            {
                return (Array.Empty<double>(), new[] { 0L });
            }
            dims.Add(length);
            probe = probe.EnumerateArray().First();
        }

        if (probe.ValueKind != JsonValueKind.Number)
        {
            return (null, Array.Empty<long>());
        }

        // Phase 2: flatten all values, validating the structure
        var flat = new List<double>();
        if (!FlattenInto(element, flat))
        {
            return (null, Array.Empty<long>());
        }

        var expected = dims.Count > 0 ? dims.Aggregate(1L, (a, b) => a * b) : 1;
        return expected == flat.Count
            ? (flat.ToArray(), dims.ToArray())
            : (null, Array.Empty<long>());
    }

    private static bool FlattenInto(JsonElement element, List<double> flat)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Number:
                flat.Add(element.GetDouble());
                return true;
            case JsonValueKind.Array:
                foreach (var item in element.EnumerateArray())
                {
                    if (!FlattenInto(item, flat))
                    {
                        return false;
                    }
                }
                return true;
            default:
                return false;
        }
    }
}
