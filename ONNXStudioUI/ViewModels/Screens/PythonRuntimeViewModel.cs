using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Core.Python;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>One choice of the interpreter list: "Automatic" (no interpreter) or a probed interpreter.</summary>
public sealed class PythonInterpreterItem
{
    public PythonInterpreter? Interpreter { get; }
    public string Display { get; }
    public string Details { get; }

    public PythonInterpreterItem(PythonInterpreter? interpreter)
    {
        Interpreter = interpreter;
        if (interpreter == null)
        {
            Display = "Automatic (managed runtime if installed, otherwise the first complete system Python)";
            Details = string.Empty;
            return;
        }

        Display = $"Python {interpreter.Version} - {interpreter.Source} - {interpreter.ExecutablePath}";
        Details = Describe(interpreter);
    }

    public static string Describe(PythonInterpreter interpreter)
    {
        if (interpreter.CanConvert)
            return $"Ready: inference and ONNX conversion (scikit-learn {interpreter.PackageVersion("scikit-learn")}, skl2onnx {interpreter.PackageVersion("skl2onnx")})";
        if (interpreter.CanRunInference)
            return $"Inference ready (scikit-learn {interpreter.PackageVersion("scikit-learn")}); ONNX conversion needs: {string.Join(", ", interpreter.MissingForConversion)}";
        return $"Missing packages: {string.Join(", ", interpreter.MissingForConversion)}";
    }
}

/// <summary>
/// Python runtime section of the settings: shows the interpreter in use, lets the
/// user pick one found on the machine or provide their own, or have ONNX Studio
/// download and install Python and the required packages in its own folder.
/// </summary>
public partial class PythonRuntimeViewModel : ObservableObject
{
    private const int MaxLogLines = 300;

    private readonly IPythonRuntimeService _runtime;
    private readonly IFilePickerService _picker;
    private readonly IToastService _toast;
    private readonly System.Collections.Generic.List<string> _logLines = new();
    private CancellationTokenSource? _operation;
    private bool _updatingList;

    public ObservableCollection<PythonInterpreterItem> Interpreters { get; } = new();

    [ObservableProperty]
    private PythonInterpreterItem? _selectedInterpreter;

    [ObservableProperty]
    private string _status = "Checking Python...";

    [ObservableProperty]
    private string? _selectedDetails;

    [ObservableProperty]
    private string _customPath = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanInstall))]
    [NotifyPropertyChangedFor(nameof(CanInstallPackages))]
    [NotifyPropertyChangedFor(nameof(CanRemove))]
    private bool _isBusy;

    [ObservableProperty]
    private string? _progressMessage;

    [ObservableProperty]
    private double _progressValue;

    [ObservableProperty]
    private bool _isProgressIndeterminate;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasLog))]
    private string _log = string.Empty;

    [ObservableProperty]
    private string? _error;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanInstallPackages))]
    [NotifyPropertyChangedFor(nameof(CanRemove))]
    [NotifyPropertyChangedFor(nameof(InstallPythonLabel))]
    private bool _isManagedInstalled;

    public bool HasLog => Log.Length > 0;
    public bool CanInstall => !IsBusy;
    public bool CanInstallPackages => !IsBusy && IsManagedInstalled;
    public bool CanRemove => !IsBusy && IsManagedInstalled;
    public string InstallPythonLabel => IsManagedInstalled ? "Reinstall Python" : "Download and install Python";

    public PythonRuntimeViewModel(IPythonRuntimeService runtime, IFilePickerService picker, IToastService toast)
    {
        _runtime = runtime;
        _picker = picker;
        _toast = toast;
    }

    /// <summary>Detects interpreters and the one currently in use.</summary>
    [RelayCommand]
    public async Task RefreshAsync()
    {
        if (IsBusy) return;
        IsBusy = true;
        IsProgressIndeterminate = true;
        ProgressMessage = "Looking for Python interpreters...";
        try
        {
            await ReloadAsync(refreshActive: true).ConfigureAwait(true);
        }
        finally
        {
            IsBusy = false;
            IsProgressIndeterminate = false;
            ProgressMessage = null;
        }
    }

    [RelayCommand]
    private async Task InstallPythonAsync()
    {
        await RunOperationAsync("Installing Python...", async (progress, token) =>
        {
            var result = await _runtime.InstallPythonAsync(progress, token).ConfigureAwait(true);
            if (result.IsFailure) { Error = result.Error!.Message; AppendLog(result.Error.TechnicalDetails); return; }
            _toast.Show("Python installed. Now install the packages.");
        }).ConfigureAwait(true);
    }

    [RelayCommand]
    private async Task InstallPackagesAsync()
    {
        await RunOperationAsync("Installing packages...", async (progress, token) =>
        {
            var result = await _runtime.InstallPackagesAsync(progress, token).ConfigureAwait(true);
            if (result.IsFailure) { Error = result.Error!.Message; AppendLog(result.Error.TechnicalDetails); return; }
            _toast.Show("Python packages installed");
        }).ConfigureAwait(true);
    }

    [RelayCommand]
    private async Task RemoveManagedAsync()
    {
        await RunOperationAsync("Removing the managed Python...", async (_, token) =>
        {
            await _runtime.RemoveManagedAsync(token).ConfigureAwait(true);
            _toast.Show("Managed Python removed");
        }).ConfigureAwait(true);
    }

    [RelayCommand]
    private void CancelOperation() => _operation?.Cancel();

    [RelayCommand]
    private async Task BrowseInterpreterAsync()
    {
        var path = await _picker.PickPythonInterpreterAsync().ConfigureAwait(true);
        if (path == null) return;
        CustomPath = path;
        await UseCustomAsync().ConfigureAwait(true);
    }

    [RelayCommand]
    private async Task BrowseEnvironmentAsync()
    {
        var path = await _picker.PickFolderAsync("Select a Python environment folder (venv, conda env, installation)").ConfigureAwait(true);
        if (path == null) return;
        CustomPath = path;
        await UseCustomAsync().ConfigureAwait(true);
    }

    /// <summary>Probes the typed / browsed interpreter or environment and starts using it.</summary>
    [RelayCommand]
    public async Task UseCustomAsync()
    {
        Error = null;
        if (string.IsNullOrWhiteSpace(CustomPath))
        {
            Error = "Enter the path of a Python interpreter or environment folder.";
            return;
        }

        IsBusy = true;
        IsProgressIndeterminate = true;
        ProgressMessage = "Checking the interpreter...";
        try
        {
            var probed = await _runtime.ProbeCustomAsync(CustomPath).ConfigureAwait(true);
            if (probed.IsFailure)
            {
                Error = probed.Error!.Message;
                return;
            }

            _runtime.Select(new PythonSelection(probed.Value!.ExecutablePath, PythonSource.Custom));
            await ReloadAsync(refreshActive: true).ConfigureAwait(true);
            _toast.Show("Using " + probed.Value.ExecutablePath);
        }
        finally
        {
            IsBusy = false;
            IsProgressIndeterminate = false;
            ProgressMessage = null;
        }
    }

    partial void OnSelectedInterpreterChanged(PythonInterpreterItem? value)
    {
        SelectedDetails = value?.Details is { Length: > 0 } details ? details : null;
        if (_updatingList || value == null) return;

        _runtime.Select(value.Interpreter == null
            ? PythonSelection.Automatic
            : new PythonSelection(value.Interpreter.ExecutablePath, value.Interpreter.Source));
        _ = RefreshStatusAsync();
    }

    private async Task RunOperationAsync(string message, Func<IProgress<PythonInstallProgress>, CancellationToken, Task> work)
    {
        if (IsBusy) return;
        Error = null;
        IsBusy = true;
        IsProgressIndeterminate = true;
        ProgressValue = 0;
        ProgressMessage = message;
        _logLines.Clear();
        Log = string.Empty;
        _operation = new CancellationTokenSource();

        var progress = new Progress<PythonInstallProgress>(p =>
        {
            ProgressMessage = p.Message.Length > 120 ? p.Message[..120] : p.Message;
            IsProgressIndeterminate = p.Percent == null;
            if (p.Percent is { } fraction) ProgressValue = fraction * 100;
            AppendLog(p.Message);
        });

        try
        {
            await work(progress, _operation.Token).ConfigureAwait(true);
        }
        catch (OperationCanceledException)
        {
            Error = "Operation cancelled.";
        }
        catch (Exception ex) when (ex is System.IO.IOException or UnauthorizedAccessException or InvalidOperationException)
        {
            Error = "The operation failed: " + ex.Message;
        }
        finally
        {
            _operation.Dispose();
            _operation = null;
            await ReloadAsync(refreshActive: true).ConfigureAwait(true);
            IsBusy = false;
            IsProgressIndeterminate = false;
            ProgressMessage = null;
        }
    }

    private async Task ReloadAsync(bool refreshActive)
    {
        IsManagedInstalled = _runtime.IsManagedInstalled;
        var found = await _runtime.DiscoverAsync().ConfigureAwait(true);
        var selection = _runtime.Selection;

        _updatingList = true;
        try
        {
            Interpreters.Clear();
            var automatic = new PythonInterpreterItem(null);
            Interpreters.Add(automatic);
            foreach (var interpreter in found) Interpreters.Add(new PythonInterpreterItem(interpreter));

            if (!selection.IsAutomatic && Interpreters.All(i => i.Interpreter == null ||
                    !string.Equals(i.Interpreter.ExecutablePath, selection.InterpreterPath, StringComparison.OrdinalIgnoreCase)))
            {
                var probed = await _runtime.ProbeCustomAsync(selection.InterpreterPath!).ConfigureAwait(true);
                if (probed.IsSuccess) Interpreters.Add(new PythonInterpreterItem(probed.Value));
            }

            SelectedInterpreter = selection.IsAutomatic
                ? automatic
                : Interpreters.FirstOrDefault(i => i.Interpreter != null &&
                      string.Equals(i.Interpreter.ExecutablePath, selection.InterpreterPath, StringComparison.OrdinalIgnoreCase)) ?? automatic;
        }
        finally
        {
            _updatingList = false;
        }

        await RefreshStatusAsync(refreshActive).ConfigureAwait(true);
    }

    private async Task RefreshStatusAsync(bool refresh = true)
    {
        var active = await _runtime.GetActiveAsync(refresh).ConfigureAwait(true);
        Status = Describe(active);
    }

    internal static string Describe(PythonInterpreter? active)
    {
        if (active == null)
        {
            return "No usable Python runtime. Install the ONNX Studio runtime below, or select an existing interpreter.";
        }

        var prefix = $"Using Python {active.Version} ({active.Source}). ";
        if (active.CanConvert) return prefix + "Ready: inference and ONNX conversion are available.";
        if (active.CanRunInference)
            return prefix + $"Inference is available; ONNX conversion needs: {string.Join(", ", active.MissingForConversion)}.";
        return prefix + $"Missing packages: {string.Join(", ", active.MissingForConversion)}." +
               (active.Source == PythonSource.Managed ? " Use 'Install packages'." : " Install them in this interpreter or select another one.");
    }

    private void AppendLog(string line)
    {
        if (string.IsNullOrWhiteSpace(line)) return;
        _logLines.Add(line.TrimEnd());
        if (_logLines.Count > MaxLogLines) _logLines.RemoveRange(0, _logLines.Count - MaxLogLines);
        Log = string.Join(Environment.NewLine, _logLines);
    }
}
