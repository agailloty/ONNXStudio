using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Services;

/// <summary>
/// Runs local inference (US-004): validates inputs against the model schema,
/// converts values to the model element type, executes ONNX Runtime and
/// returns post-processed output tensors.
/// </summary>
public interface IInferenceService
{
    Task<Result<InferenceResult, InferenceError>> RunAsync(
        OnnxModel model,
        IReadOnlyDictionary<string, InferenceInputValue> inputs,
        CancellationToken cancellationToken = default);
}

public sealed class InferenceService : IInferenceService
{
    private readonly IInferenceSessionManager _sessionManager;
    private readonly ILogger<InferenceService> _logger;
    private readonly SemaphoreSlim _concurrencyGate;

    public InferenceService(
        IInferenceSessionManager sessionManager,
        Microsoft.Extensions.Options.IOptions<Configuration.OnnxStudioOptions> options,
        ILogger<InferenceService> logger)
    {
        _sessionManager = sessionManager;
        _logger = logger;
        _concurrencyGate = new SemaphoreSlim(Math.Max(1, options.Value.MaxConcurrentInferences));
    }

    public async Task<Result<InferenceResult, InferenceError>> RunAsync(
        OnnxModel model,
        IReadOnlyDictionary<string, InferenceInputValue> inputs,
        CancellationToken cancellationToken = default)
    {
        // Validate presence of every model input
        foreach (var schema in model.Inputs)
        {
            if (!inputs.TryGetValue(schema.Name, out var value) || value.Data.Length == 0)
            {
                return Result<InferenceResult, InferenceError>.Failure(
                    new InferenceError(InferenceErrorCode.MissingInput,
                        $"The input '{schema.Name}' is required.", $"Expected {schema.ToDisplayString()}"));
            }
        }

        var validation = ValidateShapes(model, inputs);
        if (validation != null)
        {
            return Result<InferenceResult, InferenceError>.Failure(validation);
        }

        await _concurrencyGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            using var lease = _sessionManager.Acquire(model);
            var session = lease.Session;

            var ortInputs = new Dictionary<string, OrtValue>();
            try
            {
                foreach (var schema in model.Inputs)
                {
                    ortInputs[schema.Name] = CreateOrtValue(schema, inputs[schema.Name]);
                }

                var stopwatch = Stopwatch.StartNew();
                using var runOptions = new RunOptions();
                using var cancellation = cancellationToken.Register(() => runOptions.Terminate = true);
                var outputNames = session.OutputNames;
                using var rawOutputs = session.Run(runOptions, ortInputs, outputNames);
                stopwatch.Stop();

                var outputs = PostprocessOutputs(model, outputNames, rawOutputs);
                var result = new InferenceResult(model.Id, stopwatch.ElapsedMilliseconds, outputs);

                _logger.LogInformation("Inference on {Model} completed in {Ms} ms ({Outputs} outputs)",
                    model.Name, stopwatch.ElapsedMilliseconds, outputs.Count);

                return Result<InferenceResult, InferenceError>.Success(result);
            }
            finally
            {
                foreach (var ortValue in ortInputs.Values)
                {
                    ortValue.Dispose();
                }
            }
        }
        catch (OnnxRuntimeException ex)
        {
            _logger.LogError(ex, "Inference failed on {Model}", model.Name);
            return Result<InferenceResult, InferenceError>.Failure(
                new InferenceError(InferenceErrorCode.InferenceFailed,
                    "Inference failed. Check the tensor types and dimensions against the model schema.", ex.Message, ex));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected inference error on {Model}", model.Name);
            return Result<InferenceResult, InferenceError>.Failure(
                new InferenceError(InferenceErrorCode.InferenceFailed,
                    "An unexpected error occurred during inference.", ex.Message, ex));
        }
        finally
        {
            _concurrencyGate.Release();
        }
    }

    private static InferenceError? ValidateShapes(OnnxModel model, IReadOnlyDictionary<string, InferenceInputValue> inputs)
    {
        foreach (var schema in model.Inputs)
        {
            if (!inputs.TryGetValue(schema.Name, out var value))
            {
                continue;
            }

            // Scalar input is valid for any 1-element tensor
            if (value.Shape is null)
            {
                if (value.Data.Length != 1 || schema.Shape.Any(d => d is not (null or 1)))
                {
                    return new InferenceError(InferenceErrorCode.InvalidInput,
                        $"The input '{schema.Name}' expects a single value.",
                        $"Scalar with {value.Data.Length} elements");
                }
                continue;
            }

            if (value.Shape.Length != schema.Shape.Count)
            {
                return new InferenceError(InferenceErrorCode.ShapeMismatch,
                    $"The input '{schema.Name}' expects a tensor of rank {schema.Shape.Count} but received rank {value.Shape.Length}.",
                    $"Expected {schema.ToDisplayString()}, got shape [{string.Join(", ", value.Shape)}]");
            }

            for (int i = 0; i < value.Shape.Length; i++)
            {
                var expected = schema.Shape[i];
                if (value.Shape[i] <= 0 || (expected.HasValue && expected.Value != value.Shape[i]))
                {
                    return new InferenceError(InferenceErrorCode.ShapeMismatch,
                        $"The input '{schema.Name}' expects dimension {i} = {expected?.ToString() ?? "a positive size"} but received {value.Shape[i]}.",
                        $"Expected {schema.ToDisplayString()}, got shape [{string.Join(", ", value.Shape)}]");
                }
            }

            long elementCount = 1;
            foreach (var dimension in value.Shape)
            {
                if (elementCount > long.MaxValue / dimension)
                    return new InferenceError(InferenceErrorCode.InvalidInput, $"The shape of '{schema.Name}' is too large.");
                elementCount *= dimension;
            }
            if (elementCount != value.Data.Length)
            {
                return new InferenceError(InferenceErrorCode.InvalidInput,
                    $"The input '{schema.Name}' declares {elementCount} elements but {value.Data.Length} were provided.");
            }
        }
        return null;
    }

    private static OrtValue CreateOrtValue(TensorSchema schema, InferenceInputValue value)
    {
        var dimensions = value.Shape ?? schema.Shape.Select(d => d ?? 1L).ToArray();

        switch (schema.Type)
        {
            case DataType.Float16:
                return OrtValue.CreateTensorValueFromMemory(ToArray<float>(value.Data).Select(v => (Float16)v).ToArray(), dimensions);
            case DataType.String:
                return OrtValue.CreateFromStringTensor(new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<string>(
                    (string[])value.Data, dimensions.Select(d => checked((int)d)).ToArray()));
            case DataType.Uint16:
                return OrtValue.CreateTensorValueFromMemory(ToArray<ushort>(value.Data), dimensions);
            case DataType.Uint32:
                return OrtValue.CreateTensorValueFromMemory(ToArray<uint>(value.Data), dimensions);
            case DataType.Uint64:
                return OrtValue.CreateTensorValueFromMemory(ToArray<ulong>(value.Data), dimensions);
            case DataType.Float32:
                return OrtValue.CreateTensorValueFromMemory(ToArray<float>(value.Data), dimensions);
            case DataType.Float64:
                return OrtValue.CreateTensorValueFromMemory(ToArray<double>(value.Data), dimensions);
            case DataType.Int64:
                return OrtValue.CreateTensorValueFromMemory(ToArray<long>(value.Data), dimensions);
            case DataType.Int32:
                return OrtValue.CreateTensorValueFromMemory(ToArray<int>(value.Data), dimensions);
            case DataType.Int16:
                return OrtValue.CreateTensorValueFromMemory(ToArray<short>(value.Data), dimensions);
            case DataType.Int8:
                return OrtValue.CreateTensorValueFromMemory(ToArray<sbyte>(value.Data), dimensions);
            case DataType.Uint8:
                return OrtValue.CreateTensorValueFromMemory(ToArray<byte>(value.Data), dimensions);
            case DataType.Bool:
                return OrtValue.CreateTensorValueFromMemory(ToArray<bool>(value.Data), dimensions);
            default:
                throw new InvalidOperationException(
                    $"The element type {schema.Type.ToDisplayName()} of input '{schema.Name}' is not supported for inference yet.");
        }
    }

    private static T[] ToArray<T>(Array source) where T : struct
    {
        if (source is T[] typed)
        {
            return typed;
        }

        var result = new T[source.Length];
        for (int i = 0; i < source.Length; i++)
        {
            result[i] = (T)System.Convert.ChangeType(source.GetValue(i)!, typeof(T));
        }
        return result;
    }

    private static List<TensorOutput> PostprocessOutputs(
        OnnxModel model,
        IReadOnlyList<string> outputNames,
        IDisposableReadOnlyCollection<OrtValue> rawOutputs)
    {
        var schemaByName = model.Outputs.ToDictionary(o => o.Name, o => o.Type);
        var outputs = new List<TensorOutput>();
        var index = 0;

        foreach (var ortValue in rawOutputs)
        {
            var name = index < outputNames.Count ? outputNames[index] : $"output_{index}";
            var type = schemaByName.GetValueOrDefault(name, DataType.Float32);
            var shapeInfo = ortValue.GetTensorTypeAndShape();
            var shape = shapeInfo.Shape;

            switch (type)
            {
                case DataType.Float16:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetTensorDataAsSpan<Float16>().ToArray().Select(v => (float)v).ToArray()));
                    break;
                case DataType.String:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetStringTensorAsArray()));
                    break;
                case DataType.Int8:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetTensorDataAsSpan<sbyte>().ToArray()));
                    break;
                case DataType.Int16:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetTensorDataAsSpan<short>().ToArray()));
                    break;
                case DataType.Uint8:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetTensorDataAsSpan<byte>().ToArray()));
                    break;
                case DataType.Uint16:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetTensorDataAsSpan<ushort>().ToArray()));
                    break;
                case DataType.Uint32:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetTensorDataAsSpan<uint>().ToArray()));
                    break;
                case DataType.Uint64:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetTensorDataAsSpan<ulong>().ToArray()));
                    break;
                case DataType.Float64:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetTensorDataAsSpan<double>().ToArray()));
                    break;
                case DataType.Int64:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetTensorDataAsSpan<long>().ToArray()));
                    break;
                case DataType.Int32:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetTensorDataAsSpan<int>().ToArray()));
                    break;
                case DataType.Bool:
                    outputs.Add(new TensorOutput(name, type, shape, ortValue.GetTensorDataAsSpan<bool>().ToArray()));
                    break;
                default:
                    outputs.Add(new TensorOutput(name, DataType.Float32, shape, ortValue.GetTensorDataAsSpan<float>().ToArray()));
                    break;
            }
            index++;
        }

        return outputs;
    }
}
