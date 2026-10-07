using Microsoft.Extensions.Logging;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Services;

/// <summary>
/// Runs local inference (US-004): validates inputs against the model schema, then delegates the
/// execution to the <see cref="IInferenceBackend"/> that handles the model's technology.
/// </summary>
public interface IInferenceService
{
    Task<Result<InferenceResult, InferenceError>> RunAsync(
        IModel model,
        IReadOnlyDictionary<string, InferenceInputValue> inputs,
        CancellationToken cancellationToken = default);
}

public sealed class InferenceService : IInferenceService
{
    private readonly IReadOnlyList<IInferenceBackend> _backends;
    private readonly ILogger<InferenceService> _logger;
    private readonly SemaphoreSlim _concurrencyGate;

    /// <param name="additionalBackends">Backends for the other model technologies (scikit-learn, ...); ONNX Runtime is built in.</param>
    public InferenceService(
        IInferenceSessionManager sessionManager,
        Microsoft.Extensions.Options.IOptions<Configuration.OnnxStudioOptions> options,
        ILogger<InferenceService> logger,
        IEnumerable<IInferenceBackend>? additionalBackends = null)
    {
        _backends = [new OnnxInferenceBackend(sessionManager, logger), .. additionalBackends ?? []];
        _logger = logger;
        _concurrencyGate = new SemaphoreSlim(Math.Max(1, options.Value.MaxConcurrentInferences));
    }

    public async Task<Result<InferenceResult, InferenceError>> RunAsync(
        IModel model,
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

        var backend = _backends.FirstOrDefault(b => b.CanRun(model));
        if (backend == null)
        {
            return Result<InferenceResult, InferenceError>.Failure(
                new InferenceError(InferenceErrorCode.InferenceFailed, $"No inference engine can run {model.Format} models."));
        }

        await _concurrencyGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var result = await backend.RunAsync(model, inputs, cancellationToken).ConfigureAwait(false);
            if (result.IsSuccess)
            {
                _logger.LogInformation("Inference on {Model} completed in {Ms} ms ({Outputs} outputs)",
                    model.Name, result.Value!.ExecutionTimeMs, result.Value.Outputs.Count);
            }
            return result;
        }
        finally
        {
            _concurrencyGate.Release();
        }
    }

    private static InferenceError? ValidateShapes(IModel model, IReadOnlyDictionary<string, InferenceInputValue> inputs)
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

}