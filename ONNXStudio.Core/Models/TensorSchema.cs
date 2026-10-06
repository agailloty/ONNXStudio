namespace ONNXStudio.Core.Models;

/// <summary>
/// Schema of one model input or output tensor. A null dimension is a dynamic
/// dimension (e.g. batch size).
/// </summary>
public sealed class TensorSchema
{
    public string Name { get; }
    public DataType Type { get; }
    public IReadOnlyList<long?> Shape { get; }
    public string Description { get; set; } = string.Empty;

    public TensorSchema(string name, DataType type, IReadOnlyList<long?> shape)
    {
        Name = name;
        Type = type;
        Shape = shape;
    }

    public bool HasDynamicDimension => Shape.Any(d => d is null);

    /// <summary>
    /// Display form like "float32[1, 3, 224, 224]" with "?" for dynamic dims.
    /// </summary>
    public string ToDisplayString()
    {
        var dims = string.Join(", ", Shape.Select(d => d?.ToString() ?? "?"));
        return $"{Type.ToDisplayName()}[{dims}]";
    }
}
