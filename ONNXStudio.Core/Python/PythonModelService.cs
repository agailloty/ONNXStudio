using System.Diagnostics;
using System.Text.Json.Nodes;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Python;

/// <summary>
/// Inference and ONNX conversion of scikit-learn models saved with joblib or pickle.
/// Loading such a file executes code it contains, so every operation requires
/// the caller to confirm that the file is trusted.
/// </summary>
public interface IPythonModelService
{
    Task<Result<PythonModelInfo, PythonError>> InspectAsync(string modelPath, bool trustConfirmed, CancellationToken cancellationToken = default);

    Task<Result<PythonPredictionResult, PythonError>> PredictAsync(PythonPredictionRequest request, CancellationToken cancellationToken = default);

    Task<Result<PythonConversionResult, PythonError>> ConvertAsync(PythonConversionRequest request, CancellationToken cancellationToken = default);
}

public sealed class PythonModelService : IPythonModelService
{
    private readonly IPythonRuntimeService _runtime;
    private readonly IPythonWorkerClient _worker;

    public PythonModelService(IPythonRuntimeService runtime, IPythonWorkerClient worker)
    {
        _runtime = runtime;
        _worker = worker;
    }

    public async Task<Result<PythonModelInfo, PythonError>> InspectAsync(string modelPath, bool trustConfirmed, CancellationToken cancellationToken = default)
    {
        var prepared = await PrepareAsync(modelPath, trustConfirmed, conversion: false, cancellationToken).ConfigureAwait(false);
        if (prepared.IsFailure) return Result<PythonModelInfo, PythonError>.Failure(prepared.Error!);

        var response = await _worker.RunAsync(prepared.Value!, "inspect", new JsonObject { ["modelPath"] = Path.GetFullPath(modelPath) }, cancellationToken)
            .ConfigureAwait(false);
        return response.IsFailure
            ? Result<PythonModelInfo, PythonError>.Failure(response.Error!)
            : Result<PythonModelInfo, PythonError>.Success(PythonModelInfo.FromJson(response.Value!));
    }

    public async Task<Result<PythonPredictionResult, PythonError>> PredictAsync(PythonPredictionRequest request, CancellationToken cancellationToken = default)
    {
        if (request.Rows.Count == 0 || request.Rows.All(r => r.Count == 0))
        {
            return Result<PythonPredictionResult, PythonError>.Failure(new PythonError(PythonErrorCode.InvalidInput, "Enter at least one row of input values."));
        }

        var prepared = await PrepareAsync(request.ModelPath, request.TrustConfirmed, conversion: false, cancellationToken).ConfigureAwait(false);
        if (prepared.IsFailure) return Result<PythonPredictionResult, PythonError>.Failure(prepared.Error!);

        var json = new JsonObject
        {
            ["modelPath"] = Path.GetFullPath(request.ModelPath),
            ["method"] = string.IsNullOrWhiteSpace(request.Method) ? "auto" : request.Method,
            ["columns"] = request.Columns == null ? null : new JsonArray(request.Columns.Select(c => (JsonNode?)JsonValue.Create(c)).ToArray()),
            ["rows"] = new JsonArray(request.Rows
                .Select(row => (JsonNode?)new JsonArray(row.Select(v => (JsonNode?)JsonValue.Create(v)).ToArray())).ToArray())
        };

        var stopwatch = Stopwatch.StartNew();
        var response = await _worker.RunAsync(prepared.Value!, "predict", json, cancellationToken).ConfigureAwait(false);
        stopwatch.Stop();
        if (response.IsFailure) return Result<PythonPredictionResult, PythonError>.Failure(response.Error!);

        var result = response.Value!;
        var outputs = (result["outputs"] as JsonArray)?.Select(o => PythonPredictionOutput.FromJson(o!)).ToArray()
                      ?? Array.Empty<PythonPredictionOutput>();
        return Result<PythonPredictionResult, PythonError>.Success(new PythonPredictionResult(
            outputs,
            PythonModelInfo.Strings(result["classes"]),
            result["rowCount"]?.GetValue<int>() ?? request.Rows.Count,
            stopwatch.ElapsedMilliseconds,
            PythonModelInfo.Strings(result["warnings"]) ?? Array.Empty<string>()));
    }

    public async Task<Result<PythonConversionResult, PythonError>> ConvertAsync(PythonConversionRequest request, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(request.OutputPath) ||
            !string.Equals(Path.GetExtension(request.OutputPath), ".onnx", StringComparison.OrdinalIgnoreCase))
        {
            return Result<PythonConversionResult, PythonError>.Failure(new PythonError(PythonErrorCode.InvalidInput,
                "The output file must have the .onnx extension."));
        }

        if (request.TargetOpset is < 1 or > 30)
        {
            return Result<PythonConversionResult, PythonError>.Failure(new PythonError(PythonErrorCode.InvalidInput,
                "The target opset must be between 1 and 30."));
        }

        var prepared = await PrepareAsync(request.ModelPath, request.TrustConfirmed, conversion: true, cancellationToken).ConfigureAwait(false);
        if (prepared.IsFailure) return Result<PythonConversionResult, PythonError>.Failure(prepared.Error!);

        var json = new JsonObject
        {
            ["modelPath"] = Path.GetFullPath(request.ModelPath),
            ["outputPath"] = Path.GetFullPath(request.OutputPath),
            ["targetOpset"] = request.TargetOpset,
            ["zipmap"] = !request.DisableZipMap,
            ["validate"] = request.Validate,
            ["inputTypes"] = request.InputTypes is { Count: > 0 }
                ? new JsonArray(request.InputTypes.Select(t => (JsonNode?)JsonValue.Create(t)).ToArray())
                : null
        };

        var response = await _worker.RunAsync(prepared.Value!, "convert", json, cancellationToken).ConfigureAwait(false);
        return response.IsFailure
            ? Result<PythonConversionResult, PythonError>.Failure(response.Error!)
            : Result<PythonConversionResult, PythonError>.Success(PythonConversionResult.FromJson(response.Value!));
    }

    private async Task<Result<PythonInterpreter, PythonError>> PrepareAsync(
        string modelPath, bool trustConfirmed, bool conversion, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
        {
            return Fail(PythonErrorCode.FileNotFound, $"The file '{Path.GetFileName(modelPath)}' does not exist.");
        }

        if (!PythonModel.IsPythonModelFile(modelPath))
        {
            return Fail(PythonErrorCode.InvalidFormat,
                $"'{Path.GetFileName(modelPath)}' is not a joblib/pickle model ({string.Join(", ", PythonModel.SupportedExtensions)}).");
        }

        if (!trustConfirmed)
        {
            return Fail(PythonErrorCode.TrustRequired,
                "Loading a joblib/pickle file runs the code it contains. Confirm that you trust this file first.");
        }

        var interpreter = await _runtime.GetActiveAsync(cancellationToken: cancellationToken).ConfigureAwait(false);
        if (interpreter == null)
        {
            return Fail(PythonErrorCode.NoRuntime,
                "No Python runtime is available. Install one or select an existing interpreter in Settings > Python runtime.");
        }

        var missing = conversion ? interpreter.MissingForConversion : interpreter.MissingForInference;
        if (missing.Count > 0)
        {
            var hint = interpreter.Source == PythonSource.Managed
                ? "Install them from Settings > Python runtime."
                : "Install them in that interpreter (pip install " + string.Join(" ", missing) + ") or select another one in Settings.";
            return Fail(PythonErrorCode.MissingPackages,
                $"The selected Python ({interpreter.ExecutablePath}) is missing: {string.Join(", ", missing)}. {hint}");
        }

        return Result<PythonInterpreter, PythonError>.Success(interpreter);
    }

    private static Result<PythonInterpreter, PythonError> Fail(PythonErrorCode code, string message)
        => Result<PythonInterpreter, PythonError>.Failure(new PythonError(code, message));
}
