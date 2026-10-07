using System.Diagnostics;
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

public sealed class ProcessRunner : IProcessRunner
{
    public async Task<ProcessResult> RunAsync(ProcessSpec spec, CancellationToken cancellationToken = default)
    {
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
        process.StandardInput.Close();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        using var timeoutSource = new CancellationTokenSource();
        if (spec.Timeout is { } timeout) timeoutSource.CancelAfter(timeout);
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutSource.Token);

        try
        {
            await process.WaitForExitAsync(linked.Token).ConfigureAwait(false);
            // Parameterless overload flushes the asynchronous output readers.
            process.WaitForExit();
        }
        catch (OperationCanceledException)
        {
            TryKill(process);
            if (cancellationToken.IsCancellationRequested) throw;
            return new ProcessResult(-1, Snapshot(stdout), Snapshot(stderr), TimedOut: true);
        }

        return new ProcessResult(process.ExitCode, Snapshot(stdout), Snapshot(stderr), TimedOut: false);
    }

    private static string Snapshot(StringBuilder builder)
    {
        lock (builder) return builder.ToString();
    }

    private static void TryKill(Process process)
    {
        try
        {
            if (!process.HasExited) process.Kill(entireProcessTree: true);
        }
        catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception)
        {
        }
    }
}
