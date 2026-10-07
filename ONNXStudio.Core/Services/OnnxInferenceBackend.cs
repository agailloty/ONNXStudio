using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Services;

/// <summary>Runs <see cref="OnnxModel"/>s with ONNX Runtime: converts values to the model element types and post-processes the output tensors.</summary>
internal sealed class OnnxInferenceBackend : IInferenceBackend
{
    private readonly IInferenceSessionManager _sessionManager;
    private readonly ILogger _logger;

    public OnnxInferenceBackend(IInferenceSessionManager sessionManager, ILogger logger)
    {
        _sessionManager = sessionManager;
        _logger = logger;
    }

    public bool CanRun(IModel model) => model is OnnxModel;

    public Task<Result<InferenceResult, InferenceError>> RunAsync(
        IModel model,
        IReadOnlyDictionary<string, InferenceInputValue> inputs,
        CancellationToken cancellationToken = default)
    {
        var onnx = (OnnxModel)model;
        try
        {
            using var lease = _sessionManager.Acquire(onnx);
            var session = lease.Session;

            var ortInputs = new Dictionary<string, OrtValue>();
            try
            {
                foreach (var schema in onnx.Inputs)
                {
                    ortInputs[schema.Name] = CreateOrtValue(schema, inputs[schema.Name]);
                }

                var stopwatch = Stopwatch.StartNew();
                using var runOptions = new RunOptions();
                using var cancellation = cancellationToken.Register(() => runOptions.Terminate = true);
                var outputNames = session.OutputNames;
                using var rawOutputs = session.Run(runOptions, ortInputs, outputNames);
                stopwatch.Stop();

                var outputs = PostprocessOutputs(onnx, outputNames, rawOutputs);
                return Task.FromResult(Result<InferenceResult, InferenceError>.Success(
                    new InferenceResult(onnx.Id, stopwatch.ElapsedMilliseconds, outputs)));
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
            _logger.LogError(ex, "Inference failed on {Model}", onnx.Name);
            return Task.FromResult(Result<InferenceResult, InferenceError>.Failure(
                new InferenceError(InferenceErrorCode.InferenceFailed,
                    "Inference failed. Check the tensor types and dimensions against the model schema.", ex.Message, ex)));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected inference error on {Model}", onnx.Name);
            return Task.FromResult(Result<InferenceResult, InferenceError>.Failure(
                new InferenceError(InferenceErrorCode.InferenceFailed,
                    "An unexpected error occurred during inference.", ex.Message, ex)));
        }
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