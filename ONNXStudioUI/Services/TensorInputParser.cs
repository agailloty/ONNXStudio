using System.Globalization;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;

namespace ONNXStudioUI.Services;

/// <summary>Converts editable tensor values without losing integer precision.</summary>
public static class TensorInputParser
{
    public static Result<InferenceInputValue, string> Parse(TensorSchema schema, string text, string? explicitShape = null)
    {
        try
        {
            var parts = schema.Type == DataType.String
                ? new[] { text }
                : text.Split(new[] { ',', ';', ' ', '\r', '\n', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) throw new FormatException("Enter at least one value.");
            Array data = schema.Type switch
            {
                DataType.String => parts,
                DataType.Int64 => parts.Select(p => long.Parse(p, CultureInfo.InvariantCulture)).ToArray(),
                DataType.Int32 => parts.Select(p => int.Parse(p, CultureInfo.InvariantCulture)).ToArray(),
                DataType.Int16 => parts.Select(p => short.Parse(p, CultureInfo.InvariantCulture)).ToArray(),
                DataType.Int8 => parts.Select(p => sbyte.Parse(p, CultureInfo.InvariantCulture)).ToArray(),
                DataType.Uint8 => parts.Select(p => byte.Parse(p, CultureInfo.InvariantCulture)).ToArray(),
                DataType.Uint16 => parts.Select(p => ushort.Parse(p, CultureInfo.InvariantCulture)).ToArray(),
                DataType.Uint32 => parts.Select(p => uint.Parse(p, CultureInfo.InvariantCulture)).ToArray(),
                DataType.Uint64 => parts.Select(p => ulong.Parse(p, CultureInfo.InvariantCulture)).ToArray(),
                DataType.Bool => parts.Select(p => p == "1" || (p != "0" && bool.Parse(p))).ToArray(),
                _ => parts.Select(p => ParseFloat(p, schema.Type)).ToArray()
            };
            long[] shape;
            if (!string.IsNullOrWhiteSpace(explicitShape))
            {
                shape = explicitShape.Split(new[] { ',', ' ', 'x' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(p => long.Parse(p, CultureInfo.InvariantCulture)).ToArray();
            }
            else
            {
                shape = schema.Shape.Select(d => d ?? -1).ToArray();
                var dynamicCount = shape.Count(d => d == -1);
                if (dynamicCount > 1) throw new FormatException("Enter a shape for inputs with several dynamic dimensions.");
                if (dynamicCount == 1)
                {
                    var fixedCount = shape.Where(d => d != -1).Aggregate(1L, (a, b) => checked(a * b));
                    if (fixedCount <= 0 || data.Length % fixedCount != 0)
                        throw new FormatException("The number of values does not match the fixed dimensions.");
                    shape[System.Array.IndexOf(shape, -1L)] = data.Length / fixedCount;
                }
            }
            if (shape.Length != schema.Shape.Count || shape.Where((d, i) => d <= 0 || (schema.Shape[i].HasValue && schema.Shape[i] != d)).Any())
                throw new FormatException($"Shape must match {schema.Display}.");
            if (shape.Aggregate(1L, (a, b) => checked(a * b)) != data.Length)
                throw new FormatException($"The number of values does not match shape [{string.Join(", ", shape)}].");
            return Result<InferenceInputValue, string>.Success(new InferenceInputValue(data, shape));
        }
        catch (Exception ex) when (ex is FormatException or OverflowException)
        {
            return Result<InferenceInputValue, string>.Failure($"Invalid {schema.Type} input: {ex.Message}");
        }
    }

    private static double ParseFloat(string text, DataType type)
    {
        var value = double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
        if (!double.IsFinite(value) || (type == DataType.Float32 && !float.IsFinite((float)value)) || (type == DataType.Float16 && Math.Abs(value) > 65504))
            throw new FormatException("Enter a finite number within the tensor's range.");
        return value;
    }
}
