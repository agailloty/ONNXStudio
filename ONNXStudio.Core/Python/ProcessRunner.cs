using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace ONNXStudio.Core.Python;

public sealed record ProcessSpec(
    string FileName,
    IReadOnlyList<string> Arguments,
    string? WorkingDirectory = null,
    IReadOnlyDictionary<string, string?>? Environment = null,
    TimeSpan? Timeout = null,
    Action<string>? OnOutputLine = null);

public sealed record ProcessResult(int ExitCode, string StdOut, string StdErr, bool TimedOut)
{
    public bool Succeeded => ExitCode == 0 && !TimedOut;
}

/// <summary>Runs external processes (abstracted so Python services are testable).</summary>
public interface IProcessRunner
{
    /// <summary>
    /// Runs the process to completion. Cancellation or timeout kills the process tree.
    /// Throws <see cref="OperationCanceledException"/> on cancellation and
    /// <see cref="System.ComponentModel.Win32Exception"/> when the executable cannot be started.
    /// </summary>
    Task<ProcessResult> RunAsync(ProcessSpec spec, CancellationToken cancellationToken = default);
}

public sealed class ProcessRunner : IProcessRunner, IDisposable
{
    private readonly ConcurrentDictionary<long, Process> _activeProcesses = new();
    private readonly object _shutdownLock = new();
    private readonly CancellationTokenSource _shutdown = new();
    private long _nextId;
    private int _disposed;

    public async Task<ProcessResult> RunAsync(ProcessSpec spec, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        var startInfo = new ProcessStartInfo(spec.FileName)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        foreach (var argument in spec.Arguments) startInfo.ArgumentList.Add(argument);
        if (spec.WorkingDirectory != null) startInfo.WorkingDirectory = spec.WorkingDirectory;
        if (spec.Environment != null)
        {
            foreach (var (key, value) in spec.Environment)
            {
                if (value == null) startInfo.Environment.Remove(key);
                else startInfo.Environment[key] = value;
            }
        }

        using var process = new Process { StartInfo = startInfo };
        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data == null) return;
            lock (stdout) stdout.AppendLine(e.Data);
            spec.OnOutputLine?.Invoke(e.Data);
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data == null) return;
            lock (stderr) stderr.AppendLine(e.Data);
            spec.OnOutputLine?.Invoke(e.Data);
        };

        process.Start();
        var processId = Interlocked.Increment(ref _nextId);
        _activeProcesses.TryAdd(processId, process);
        try
        {
            process.StandardInput.Close();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            using var timeoutSource = new CancellationTokenSource();
            if (spec.Timeout is { } timeout) timeoutSource.CancelAfter(timeout);

            CancellationTokenSource linked;
            lock (_shutdownLock)
            {
                if (_disposed != 0)
                {
                    TryKill(process);
                    throw new ObjectDisposedException(nameof(ProcessRunner));
                }

                linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutSource.Token, _shutdown.Token);
            }

            using (linked)
            {
                try
                {
                    await process.WaitForExitAsync(linked.Token).ConfigureAwait(false);
                    // Parameterless overload flushes the asynchronous output readers.
                    process.WaitForExit();
                }
                catch (OperationCanceledException)
                {
                    TryKill(process);
                    if (cancellationToken.IsCancellationRequested || Volatile.Read(ref _disposed) != 0) throw;
                    return new ProcessResult(-1, Snapshot(stdout), Snapshot(stderr), TimedOut: true);
                }
            }

            return new ProcessResult(process.ExitCode, Snapshot(stdout), Snapshot(stderr), TimedOut: false);
        }
        finally
        {
            _activeProcesses.TryRemove(processId, out _);
        }
    }

    public void Dispose()
    {
        lock (_shutdownLock)
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            _shutdown.Cancel();
            _shutdown.Dispose();
        }

        foreach (var entry in _activeProcesses)
        {
            TryKill(entry.Value);
        }
    }

    private static string Snapshot(StringBuilder builder)
    {
        lock (builder) return builder.ToString();
    }

    private static void TryKill(Process process)
    {
        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit(500);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ExternalException or ObjectDisposedException or AggregateException)
        {
        }
    }
}
