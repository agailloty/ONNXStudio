using System.Text.Json.Nodes;
using Microsoft.Extensions.Logging;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Python;

/// <summary>Runs the bundled worker script with a given interpreter.</summary>
public interface IPythonWorkerClient
{
    Task<Result<JsonObject, PythonError>> RunAsync(
        PythonInterpreter interpreter, string command, JsonObject request, CancellationToken cancellationToken = default);
}

public sealed class PythonWorkerClient : IPythonWorkerClient
{
    private const string ScriptName = "onnxstudio_worker.py";

    private readonly PythonPaths _paths;
    private readonly IProcessRunner _runner;
    private readonly OnnxStudioPythonOptions _options;
    private readonly ILogger<PythonWorkerClient> _logger;
    private readonly object _scriptLock = new();

    public PythonWorkerClient(PythonPaths paths, IProcessRunner runner, OnnxStudioPythonOptions options, ILogger<PythonWorkerClient> logger)
    {
        _paths = paths;
        _runner = runner;
        _options = options;
        _logger = logger;
    }

    public async Task<Result<JsonObject, PythonError>> RunAsync(
        PythonInterpreter interpreter, string command, JsonObject request, CancellationToken cancellationToken = default)
    {
        string script;
        try
        {
            script = EnsureScript();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return Fail(PythonErrorCode.WorkerFailed, "The Python worker script could not be written. Check folder permissions.", ex.Message);
        }

        var work = Path.Combine(Path.GetTempPath(), "onnxstudio-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(work);
            var requestPath = Path.Combine(work, "request.json");
            var responsePath = Path.Combine(work, "response.json");
            await File.WriteAllTextAsync(requestPath, request.ToJsonString(), cancellationToken).ConfigureAwait(false);

            ProcessResult process;
            try
            {
                process = await _runner.RunAsync(new ProcessSpec(
                    interpreter.ExecutablePath,
                    new[] { script, command, requestPath, responsePath },
                    WorkingDirectory: work,
                    Environment: PythonEnvironment.For(interpreter.Source),
                    Timeout: TimeSpan.FromSeconds(Math.Max(10, _options.WorkerTimeoutSeconds))), cancellationToken).ConfigureAwait(false);
            }
            catch (System.ComponentModel.Win32Exception ex)
            {
                return Fail(PythonErrorCode.InvalidInterpreter, "The selected Python interpreter could not be started.", ex.Message);
            }

            if (process.TimedOut)
            {
                return Fail(PythonErrorCode.Timeout,
                    $"The Python operation did not finish within {_options.WorkerTimeoutSeconds} seconds and was stopped.");
            }

            if (!File.Exists(responsePath))
            {
                _logger.LogError("Python worker produced no response (exit {Code}): {Error}", process.ExitCode, process.StdErr);
                return Fail(PythonErrorCode.WorkerFailed,
                    "The Python process ended without a result (see technical details).", Tail(process.StdErr + process.StdOut));
            }

            JsonObject? response;
            try
            {
                response = JsonNode.Parse(await File.ReadAllTextAsync(responsePath, cancellationToken).ConfigureAwait(false)) as JsonObject;
            }
            catch (System.Text.Json.JsonException ex)
            {
                return Fail(PythonErrorCode.WorkerFailed, "The Python worker returned an unreadable result.", ex.Message);
            }

            if (response == null) return Fail(PythonErrorCode.WorkerFailed, "The Python worker returned an unreadable result.");

            if (response["ok"]?.GetValue<bool>() == true && response["result"] is JsonObject result)
            {
                return Result<JsonObject, PythonError>.Success(result);
            }

            var error = response["error"] as JsonObject;
            var code = MapCode(error?["code"]?.GetValue<string>());
            var message = error?["message"]?.GetValue<string>() ?? "The Python worker failed.";
            var trace = error?["trace"]?.GetValue<string>() ?? string.Empty;
            _logger.LogWarning("Python worker {Command} failed ({Code}): {Message}", command, code, message);
            return Fail(code, message, trace);
        }
        finally
        {
            try { Directory.Delete(work, recursive: true); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }
    }

    private static PythonErrorCode MapCode(string? code) => code switch
    {
        "file_not_found" => PythonErrorCode.FileNotFound,
        "missing_module" => PythonErrorCode.MissingModule,
        "load_failed" => PythonErrorCode.LoadFailed,
        "invalid_input" => PythonErrorCode.InvalidInput,
        "unsupported_model" => PythonErrorCode.UnsupportedModel,
        "convert_failed" => PythonErrorCode.ConversionFailed,
        "predict_failed" => PythonErrorCode.PredictionFailed,
        _ => PythonErrorCode.WorkerFailed
    };

    /// <summary>Writes the embedded worker script to disk (rewritten only when it changed).</summary>
    private string EnsureScript()
    {
        var content = ReadEmbeddedScript();
        var target = Path.Combine(_paths.WorkerDirectory, ScriptName);
        lock (_scriptLock)
        {
            if (!File.Exists(target) || File.ReadAllText(target) != content)
            {
                Directory.CreateDirectory(_paths.WorkerDirectory);
                var temp = target + ".tmp";
                File.WriteAllText(temp, content, new System.Text.UTF8Encoding(false));
                File.Move(temp, target, overwrite: true);
            }
        }
        return target;
    }

    private static string ReadEmbeddedScript()
    {
        using var stream = typeof(PythonWorkerClient).Assembly.GetManifestResourceStream(ScriptName)
                           ?? throw new InvalidOperationException("Embedded worker script not found.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd().Replace("\r\n", "\n");
    }

    private static string Tail(string text) => text.Length <= 3000 ? text : text[^3000..];

    private static Result<JsonObject, PythonError> Fail(PythonErrorCode code, string message, string details = "")
        => Result<JsonObject, PythonError>.Failure(new PythonError(code, message, details));
}
