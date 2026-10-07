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
        try
        {
            var leaves = new List<JsonElement>();
            var shape = Collect(element, leaves);
            if (shape.Length == 1 && schema.Shape.Count != 1 && !schema.HasDynamicDimension)
                shape = schema.Shape.Select(d => d!.Value).ToArray();
            if (shape.Length == 0)
                shape = schema.Shape.Select(d => d ?? 1).ToArray();
            Array data = schema.Type switch
            {
                DataType.String => leaves.Select(e => e.GetString() ?? "").ToArray(),
                DataType.Bool => leaves.Select(e => e.GetBoolean()).ToArray(),
                DataType.Int64 => leaves.Select(e => e.GetInt64()).ToArray(),
                DataType.Int32 => leaves.Select(e => e.GetInt32()).ToArray(),
                DataType.Int16 => leaves.Select(e => e.GetInt16()).ToArray(),
                DataType.Int8 => leaves.Select(e => e.GetSByte()).ToArray(),
                DataType.Uint8 => leaves.Select(e => e.GetByte()).ToArray(),
                DataType.Uint16 => leaves.Select(e => e.GetUInt16()).ToArray(),
                DataType.Uint32 => leaves.Select(e => e.GetUInt32()).ToArray(),
                DataType.Uint64 => leaves.Select(e => e.GetUInt64()).ToArray(),
                _ => leaves.Select(e => e.GetDouble()).ToArray()
            };
            return Result<InferenceInputValue, string>.Success(new InferenceInputValue(data, shape));
        }
        catch (Exception ex) when (ex is InvalidOperationException or FormatException or OverflowException)
        {
            return Result<InferenceInputValue, string>.Failure($"Expected a rectangular tensor of {schema.Type.ToDisplayName()} values. {ex.Message}");
        }
    }

    private static long[] Collect(JsonElement element, List<JsonElement> leaves)
    {
        if (element.ValueKind != JsonValueKind.Array)
        {
            leaves.Add(element);
            return Array.Empty<long>();
        }
        long[]? childShape = null;
        foreach (var child in element.EnumerateArray())
        {
            var current = Collect(child, leaves);
            if (childShape != null && !childShape.SequenceEqual(current))
                throw new FormatException("Nested array dimensions must be consistent.");
            childShape = current;
        }
        return new[] { (long)element.GetArrayLength() }.Concat(childShape ?? Array.Empty<long>()).ToArray();
    }
}
