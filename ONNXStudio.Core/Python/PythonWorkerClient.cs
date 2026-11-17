using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Nodes;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Python;

/// <summary>Runs the bundled worker script with a given interpreter.</summary>
public interface IPythonWorkerClient
{
    Task<Result<JsonObject, PythonError>> RunAsync(
        PythonInterpreter interpreter, string command, JsonObject request, CancellationToken cancellationToken = default);

    /// <summary>Starts the persistent worker process and preloads libraries so subsequent operations are fast.</summary>
    Task WarmupAsync(PythonInterpreter interpreter, CancellationToken cancellationToken = default) => Task.CompletedTask;
}

public sealed class PythonWorkerClient : IPythonWorkerClient, IDisposable
{
    private const string ScriptName = "onnxstudio_worker.py";
    private const string ReadySentinel = "__ONNXSTUDIO_READY__";
    private const string DoneSentinel = "__ONNXSTUDIO_DONE__";

    private readonly PythonPaths _paths;
    private readonly IProcessRunner _runner;
    private readonly OnnxStudioPythonOptions _options;
    private readonly ILogger<PythonWorkerClient> _logger;
    private readonly object _scriptLock = new();
    private readonly object _disposeLock = new();
    private readonly SemaphoreSlim _daemonLock = new(1, 1);
    private readonly CancellationTokenSource _disposeCts = new();
    private DaemonSession? _daemon;
    private int _disposed;

    public PythonWorkerClient(PythonPaths paths, IProcessRunner runner, OnnxStudioPythonOptions options, ILogger<PythonWorkerClient> logger)
    {
        _paths = paths;
        _runner = runner;
        _options = options;
        _logger = logger;
    }

    public async Task WarmupAsync(PythonInterpreter interpreter, CancellationToken cancellationToken = default)
    {
        if (_runner is not ProcessRunner || Volatile.Read(ref _disposed) != 0) return;

        string scriptPath, scriptContent;
        try
        {
            (scriptPath, scriptContent) = EnsureScript();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return;
        }

        try
        {
            await _daemonLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is OperationCanceledException or ObjectDisposedException)
        {
            return;
        }

        try
        {
            if (Volatile.Read(ref _disposed) != 0) return;
            await EnsureDaemonAsync(interpreter, scriptPath, scriptContent, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex) when (ex is Win32Exception or IOException or InvalidOperationException or ObjectDisposedException)
        {
            _logger.LogDebug(ex, "Python worker warmup failed");
        }
        finally
        {
            ReleaseDaemonLock();
        }
    }

    public async Task<Result<JsonObject, PythonError>> RunAsync(
        PythonInterpreter interpreter, string command, JsonObject request, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();

        string scriptPath, scriptContent;
        try
        {
            (scriptPath, scriptContent) = EnsureScript();
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

            if (_runner is ProcessRunner)
            {
                return await RunViaDaemonAsync(interpreter, scriptPath, scriptContent, command, requestPath, responsePath, cancellationToken)
                    .ConfigureAwait(false);
            }

            return await RunSingleShotAsync(interpreter, scriptPath, command, work, requestPath, responsePath, cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            try { Directory.Delete(work, recursive: true); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }
    }

    private async Task<Result<JsonObject, PythonError>> RunSingleShotAsync(
        PythonInterpreter interpreter,
        string script,
        string command,
        string work,
        string requestPath,
        string responsePath,
        CancellationToken cancellationToken)
    {
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
        catch (Win32Exception ex)
        {
            return Fail(PythonErrorCode.InvalidInterpreter, "The selected Python interpreter could not be started.", ex.Message);
        }

        if (process.TimedOut)
        {
            return Fail(PythonErrorCode.Timeout,
                $"The Python operation did not finish within {_options.WorkerTimeoutSeconds} seconds and was stopped.");
        }

        return await ParseResponseFileAsync(responsePath, process.StdErr + process.StdOut, process.ExitCode, command, cancellationToken)
            .ConfigureAwait(false);
    }

    private async Task<Result<JsonObject, PythonError>> RunViaDaemonAsync(
        PythonInterpreter interpreter,
        string scriptPath,
        string scriptContent,
        string command,
        string requestPath,
        string responsePath,
        CancellationToken cancellationToken)
    {
        await _daemonLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

            var payload = new JsonObject
            {
                ["command"] = command,
                ["requestPath"] = requestPath,
                ["responsePath"] = responsePath
            }.ToJsonString();

            for (var attempt = 0; attempt < 2; attempt++)
            {
                DaemonEnsureResult ensured;
                try
                {
                    ensured = await EnsureDaemonAsync(interpreter, scriptPath, scriptContent, cancellationToken).ConfigureAwait(false);
                }
                catch (Win32Exception ex)
                {
                    return Fail(PythonErrorCode.InvalidInterpreter, "The selected Python interpreter could not be started.", ex.Message);
                }

                if (ensured.Failure is { } failure)
                {
                    return failure;
                }

                var session = ensured.Session!;
                session.DrainOutput();

                try
                {
                    await session.Stdin.WriteLineAsync(payload.AsMemory(), cancellationToken).ConfigureAwait(false);
                    await session.Stdin.FlushAsync(cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex) when (ex is IOException or ObjectDisposedException)
                {
                    ResetDaemon();
                    if (attempt == 0 && !cancellationToken.IsCancellationRequested && Volatile.Read(ref _disposed) == 0)
                    {
                        continue;
                    }
                    return Fail(PythonErrorCode.WorkerFailed, "The Python worker process stopped unexpectedly.", ex.Message);
                }

                using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(Math.Max(10, _options.WorkerTimeoutSeconds)));
                using var linked = CreateLinkedToken(cancellationToken, timeoutCts.Token);
                bool completed;
                try
                {
                    completed = await WaitForSentinelAsync(session, DoneSentinel, linked.Token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    ResetDaemon();
                    if (cancellationToken.IsCancellationRequested || Volatile.Read(ref _disposed) != 0) throw;
                    return Fail(PythonErrorCode.Timeout,
                        $"The Python operation did not finish within {_options.WorkerTimeoutSeconds} seconds and was stopped.");
                }

                if (!completed)
                {
                    var errTail = session.ReadStderrTail();
                    var exitCode = session.TryGetExitCode();
                    ResetDaemon();
                    if (File.Exists(responsePath))
                    {
                        return await ParseResponseFileAsync(responsePath, errTail, exitCode, command, cancellationToken).ConfigureAwait(false);
                    }
                    if (attempt == 0 && errTail.Length == 0 && !cancellationToken.IsCancellationRequested && Volatile.Read(ref _disposed) == 0)
                    {
                        continue;
                    }
                    return await ParseResponseFileAsync(responsePath, errTail, exitCode, command, cancellationToken).ConfigureAwait(false);
                }

                return await ParseResponseFileAsync(responsePath, session.ReadStderrTail(), 0, command, cancellationToken).ConfigureAwait(false);
            }

            return Fail(PythonErrorCode.WorkerFailed, "The Python worker process stopped unexpectedly.");
        }
        finally
        {
            ReleaseDaemonLock();
        }
    }

    private readonly record struct DaemonEnsureResult(DaemonSession? Session, Result<JsonObject, PythonError>? Failure);

    private async Task<DaemonEnsureResult> EnsureDaemonAsync(
        PythonInterpreter interpreter, string scriptPath, string scriptContent, CancellationToken cancellationToken)
    {
        if (_daemon != null && _daemon.Matches(interpreter, scriptContent) && !_daemon.HasExited)
        {
            return new DaemonEnsureResult(_daemon, null);
        }

        var startupTimeout = TimeSpan.FromSeconds(Math.Max(60, _options.WorkerTimeoutSeconds));
        using var timeoutCts = new CancellationTokenSource(startupTimeout);
        using var linked = CreateLinkedToken(cancellationToken, timeoutCts.Token);
        linked.Token.ThrowIfCancellationRequested();

        ResetDaemon();

        var session = StartDaemonProcess(interpreter, scriptPath, scriptContent);
        _daemon = session;

        bool ready;
        try
        {
            ready = await WaitForSentinelAsync(session, ReadySentinel, linked.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            ResetDaemon();
            if (cancellationToken.IsCancellationRequested || Volatile.Read(ref _disposed) != 0) throw;
            return new DaemonEnsureResult(null, Fail(PythonErrorCode.Timeout,
                $"The Python worker did not finish starting within {(int)startupTimeout.TotalSeconds} seconds and was stopped."));
        }

        if (!ready)
        {
            var errTail = session.ReadStderrTail();
            var exitCode = session.TryGetExitCode();
            ResetDaemon();
            _logger.LogError("Python daemon failed to start (exit {Code}): {Error}", exitCode, errTail);
            return new DaemonEnsureResult(null, Fail(PythonErrorCode.WorkerFailed,
                "The Python process ended without a result (see technical details).", Tail(errTail)));
        }

        return new DaemonEnsureResult(session, null);
    }

    private DaemonSession StartDaemonProcess(PythonInterpreter interpreter, string scriptPath, string scriptContent)
    {
        var startInfo = new ProcessStartInfo(interpreter.ExecutablePath)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardInputEncoding = new UTF8Encoding(false),
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
            WorkingDirectory = Path.GetDirectoryName(scriptPath) ?? _paths.WorkerDirectory
        };
        startInfo.ArgumentList.Add("-u");
        startInfo.ArgumentList.Add(scriptPath);
        startInfo.ArgumentList.Add("--daemon");
        foreach (var (key, value) in PythonEnvironment.For(interpreter.Source))
        {
            if (value == null) startInfo.Environment.Remove(key);
            else startInfo.Environment[key] = value;
        }

        var process = new Process { StartInfo = startInfo };
        var outputLines = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });
        var session = new DaemonSession(interpreter.ExecutablePath, interpreter.Source, scriptContent, process, outputLines);
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data == null) outputLines.Writer.TryComplete();
            else outputLines.Writer.TryWrite(e.Data);
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data != null) session.AppendStderr(e.Data);
        };

        try
        {
            process.Start();
            session.AttachStdin(process.StandardInput);
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            return session;
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    private static async Task<bool> WaitForSentinelAsync(DaemonSession session, string sentinel, CancellationToken cancellationToken)
    {
        while (await session.OutputLines.Reader.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
        {
            while (session.OutputLines.Reader.TryRead(out var line))
            {
                if (string.Equals(line.Trim(), sentinel, StringComparison.Ordinal))
                {
                    return true;
                }
            }
        }
        return false;
    }

    private async Task<Result<JsonObject, PythonError>> ParseResponseFileAsync(
        string responsePath, string stdOutput, int exitCode, string command, CancellationToken cancellationToken)
    {
        if (!File.Exists(responsePath))
        {
            _logger.LogError("Python worker produced no response (exit {Code}): {Error}", exitCode, stdOutput);
            return Fail(PythonErrorCode.WorkerFailed,
                "The Python process ended without a result (see technical details).", Tail(stdOutput));
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

    private void ResetDaemon()
    {
        var current = Interlocked.Exchange(ref _daemon, null);
        current?.Dispose();
    }

    private void ReleaseDaemonLock()
    {
        try { _daemonLock.Release(); }
        catch (ObjectDisposedException) { }
    }

    private CancellationTokenSource CreateLinkedToken(CancellationToken cancellationToken, CancellationToken timeoutToken)
    {
        lock (_disposeLock)
        {
            ObjectDisposedException.ThrowIf(_disposed != 0, this);
            return CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutToken, _disposeCts.Token);
        }
    }

    public void Dispose()
    {
        lock (_disposeLock)
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            _disposeCts.Cancel();
            _disposeCts.Dispose();
        }

        ResetDaemon();
        _daemonLock.Dispose();
    }

    private sealed class DaemonSession : IDisposable
    {
        private readonly StringBuilder _stderr = new();
        private readonly object _stderrLock = new();
        private int _disposed;

        public string ExecutablePath { get; }
        public PythonSource Source { get; }
        public string ScriptContent { get; }
        public Process Process { get; }
        public StreamWriter Stdin { get; private set; } = null!;
        public Channel<string> OutputLines { get; }

        public DaemonSession(string executablePath, PythonSource source, string scriptContent, Process process, Channel<string> outputLines)
        {
            ExecutablePath = executablePath;
            Source = source;
            ScriptContent = scriptContent;
            Process = process;
            OutputLines = outputLines;
        }

        public void AttachStdin(StreamWriter stdin) => Stdin = stdin;

        public bool HasExited
        {
            get
            {
                try { return Process.HasExited; }
                catch (Exception ex) when (ex is InvalidOperationException or ObjectDisposedException) { return true; }
            }
        }

        public bool Matches(PythonInterpreter interpreter, string scriptContent)
            => Source == interpreter.Source
               && string.Equals(ExecutablePath, interpreter.ExecutablePath,
                   OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal)
               && string.Equals(ScriptContent, scriptContent, StringComparison.Ordinal);

        public void DrainOutput()
        {
            while (OutputLines.Reader.TryRead(out _)) { }
        }

        public void AppendStderr(string line)
        {
            lock (_stderrLock)
            {
                if (_stderr.Length > 16000) _stderr.Remove(0, _stderr.Length - 8000);
                _stderr.AppendLine(line);
            }
        }

        public string ReadStderrTail()
        {
            lock (_stderrLock) return Tail(_stderr.ToString());
        }

        public int TryGetExitCode()
        {
            try { return Process.HasExited ? Process.ExitCode : -1; }
            catch (Exception ex) when (ex is InvalidOperationException or ObjectDisposedException) { return -1; }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;

            try { Stdin?.Dispose(); }
            catch (Exception ex) when (ex is IOException or ObjectDisposedException) { }

            try
            {
                if (!Process.HasExited)
                {
                    Process.Kill(entireProcessTree: true);
                    Process.WaitForExit(500);
                }
            }
            catch (Exception ex) when (ex is InvalidOperationException or ExternalException or ObjectDisposedException or AggregateException)
            {
            }

            try { Process.Dispose(); }
            catch (ObjectDisposedException) { }
            OutputLines.Writer.TryComplete();
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
    private (string Path, string Content) EnsureScript()
    {
        var content = ReadEmbeddedScript();
        var target = Path.Combine(_paths.WorkerDirectory, ScriptName);
        lock (_scriptLock)
        {
            if (!File.Exists(target) || File.ReadAllText(target) != content)
            {
                Directory.CreateDirectory(_paths.WorkerDirectory);
                var temp = target + ".tmp";
                File.WriteAllText(temp, content, new UTF8Encoding(false));
                File.Move(temp, target, overwrite: true);
            }
        }
        return (target, content);
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
