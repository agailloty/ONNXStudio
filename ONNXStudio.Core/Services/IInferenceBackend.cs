using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Services;

/// <summary>
/// Runs inference for one family of models (ONNX Runtime, the Python worker, later ML.NET...).
/// <see cref="InferenceService"/> validates the inputs against the model schema, then hands them
/// to the first backend that can run the model.
/// </summary>
public interface IInferenceBackend
{
    bool CanRun(IModel model);

    Task<Result<InferenceResult, InferenceError>> RunAsync(
        IModel model,
        IReadOnlyDictionary<string, InferenceInputValue> inputs,
        CancellationToken cancellationToken = default);
}