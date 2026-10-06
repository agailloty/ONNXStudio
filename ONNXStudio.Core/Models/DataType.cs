namespace ONNXStudio.Core.Models;

/// <summary>
/// ONNX element types supported by the studio (subset of onnx.DataType).
/// </summary>
public enum DataType
{
    Undefined,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
    String
}

public static class DataTypeExtensions
{
    /// <summary>
    /// Maps an ONNX Runtime tensor element type (int) to the studio DataType.
    /// </summary>
    public static DataType FromOnnxRuntimeElementType(int elementType) => elementType switch
    {
        1 => DataType.Float32,
        2 => DataType.Uint8,
        3 => DataType.Int8,
        4 => DataType.Uint16,
        5 => DataType.Int16,
        6 => DataType.Int32,
        7 => DataType.Int64,
        8 => DataType.String,
        9 => DataType.Bool,
        10 => DataType.Float16,
        11 => DataType.Float64,
        12 => DataType.Uint32,
        13 => DataType.Uint64,
        _ => DataType.Undefined
    };

    /// <summary>
    /// Maps an ONNX Runtime tensor element type to the System.Type used for .NET arrays.
    /// </summary>
    public static Type ToClrType(this DataType dataType) => dataType switch
    {
        DataType.Float32 => typeof(float),
        DataType.Float64 => typeof(double),
        DataType.Float16 => typeof(float),
        DataType.Int8 => typeof(sbyte),
        DataType.Int16 => typeof(short),
        DataType.Int32 => typeof(int),
        DataType.Int64 => typeof(long),
        DataType.Uint8 => typeof(byte),
        DataType.Uint16 => typeof(ushort),
        DataType.Uint32 => typeof(uint),
        DataType.Uint64 => typeof(ulong),
        DataType.Bool => typeof(bool),
        DataType.String => typeof(string),
        _ => typeof(object)
    };

    /// <summary>
    /// Short ONNX-style name ("float32", "int64", ...) for display.
    /// </summary>
    public static string ToDisplayName(this DataType dataType) => dataType switch
    {
        DataType.Float32 => "float32",
        DataType.Float64 => "float64",
        DataType.Float16 => "float16",
        DataType.Int8 => "int8",
        DataType.Int16 => "int16",
        DataType.Int32 => "int32",
        DataType.Int64 => "int64",
        DataType.Uint8 => "uint8",
        DataType.Uint16 => "uint16",
        DataType.Uint32 => "uint32",
        DataType.Uint64 => "uint64",
        DataType.Bool => "bool",
        DataType.String => "string",
        _ => "undefined"
    };
}
