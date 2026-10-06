using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Services;

/// <summary>
/// A validated input value ready for inference. The array is converted to the
/// model element type by the InferenceService.
/// </summary>
public sealed class InferenceInputValue
{
    /// <summary>Shape of the tensor to build (null for a rank-0 scalar).</summary>
    public long[]? Shape { get; }

    /// <summary>Flat data (double[], long[], float[], string[]...).</summary>
    public Array Data { get; }

    public InferenceInputValue(Array data, long[]? shape)
    {
        Data = data;
        Shape = shape;
    }

    public static InferenceInputValue Scalar(double value)
        => new(new[] { value }, null);

    public static InferenceInputValue Scalars(params double[] values)
        => new(values, new[] { (long)values.Length });

    public static InferenceInputValue Array(double[] values, long[] shape)
        => new(values, shape);

    public static InferenceInputValue Tensor(float[] values, long[] shape)
        => new(values, shape);

    public static InferenceInputValue Integers(long[] values, long[] shape)
        => new(values, shape);
}

public enum InferenceErrorCode
{
    None,
    ModelNotLoaded,
    MissingInput,
    ShapeMismatch,
    TypeMismatch,
    InvalidInput,
    InferenceFailed
}

public sealed class InferenceError
{
    public InferenceErrorCode Code { get; }
    public string Message { get; }
    public string TechnicalDetails { get; }
    public Exception? OriginalException { get; }

    public InferenceError(InferenceErrorCode code, string message, string technicalDetails = "", Exception? ex = null)
    {
        Code = code;
        Message = message;
        TechnicalDetails = technicalDetails;
        OriginalException = ex;
    }

    public override string ToString() => $"[{Code}] {Message}";
}

/// <summary>
/// One output tensor of an inference run.
/// </summary>
public sealed class TensorOutput
{
    public string Name { get; }
    public DataType Type { get; }
    public IReadOnlyList<long> Shape { get; }
    public Array Data { get; }

    public TensorOutput(string name, DataType type, IReadOnlyList<long> shape, Array data)
    {
        Name = name;
        Type = type;
        Shape = shape;
        Data = data;
    }

    public long ElementCount => Data.Length;
}

/// <summary>
/// Result of a successful inference run.
/// </summary>
public sealed class InferenceResult
{
    public string ModelId { get; }
    public long ExecutionTimeMs { get; }
    public IReadOnlyList<TensorOutput> Outputs { get; }

    public InferenceResult(string modelId, long executionTimeMs, IReadOnlyList<TensorOutput> outputs)
    {
        ModelId = modelId;
        ExecutionTimeMs = executionTimeMs;
        Outputs = outputs;
    }
}
