using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Core.Python;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>One array returned by the model (predict, predict_proba...), one line per input row.</summary>
public sealed class PythonOutputItem
{
    public string Name { get; }
    public string Header { get; }
    public IReadOnlyList<string> Lines { get; }

    public PythonOutputItem(PythonPredictionOutput output, IReadOnlyList<string>? classes)
    {
        Name = output.Name;
        var shape = "[" + string.Join(", ", output.Shape) + "]";
        Header = output.Name == "predict_proba" && classes is { Count: > 0 }
            ? $"{output.DType} {shape} - columns: {string.Join(", ", classes)}"
            : $"{output.DType} {shape}";
        Lines = output.Rows.Select((value, index) => $"{index + 1}: {value}").ToArray();
        if (output.Truncated) Lines = Lines.Append("... (truncated)").ToArray();
    }
}

/// <summary>
/// Screen of a joblib / pickle model: load it (after the user confirms trust),
/// run inference on typed values and convert it to ONNX.
/// </summary>
public partial class PythonModelViewModel : ViewModelBase, IDisposable
{
    private static readonly string[] MethodChoices = { "auto", "predict", "predict_proba", "predict_log_proba", "decision_function", "transform", "all" };

    private readonly MainWindowViewModel _shell;
    private readonly IPythonModelService _service;
    private readonly IPythonRuntimeService _runtime;
    private readonly IFilePickerService _picker;
    private readonly IToastService _toast;
    private readonly Func<IModelLoadCoordinator?> _coordinator;
    private CancellationTokenSource? _operation;

    public PythonModel Model { get; }

    public IReadOnlyList<string> Methods => MethodChoices;

    // ----- runtime & trust -----

    [ObservableProperty]
    private string _runtimeStatus = "Checking Python...";

    [ObservableProperty]
    private bool _runtimeReady;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanLoad))]
    private bool _isTrusted;

    // ----- loaded model -----

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasInfo))]
    [NotifyPropertyChangedFor(nameof(IsLoaded))]
    [NotifyPropertyChangedFor(nameof(CanRun))]
    [NotifyPropertyChangedFor(nameof(CanConvert))]
    [NotifyPropertyChangedFor(nameof(LoadButtonText))]
    private PythonModelInfo? _info;

    [ObservableProperty]
    private string? _loadError;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanLoad))]
    [NotifyPropertyChangedFor(nameof(CanRun))]
    [NotifyPropertyChangedFor(nameof(CanConvert))]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private bool _isWorking;

    public bool HasInfo => Info != null;
    public bool IsLoaded => Info != null;
    public string LoadButtonText => IsLoaded ? "Reload model" : "Load model";
    public bool IsIdle => !IsWorking;
    public bool CanLoad => IsTrusted && !IsWorking;
    public bool CanRun => IsLoaded && !IsWorking;
    public bool CanConvert => IsLoaded && !IsWorking;

    public string SummaryText => Info?.Summary ?? string.Empty;
    public string FeaturesText { get; private set; } = string.Empty;
    public string ClassesText { get; private set; } = string.Empty;
    public string MethodsText { get; private set; } = string.Empty;
    public string VersionText { get; private set; } = string.Empty;
    public ObservableCollection<string> Parameters { get; } = new();
    public ObservableCollection<string> Warnings { get; } = new();

    // ----- inference -----

    [ObservableProperty]
    private string _inputText = string.Empty;

    [ObservableProperty]
    private bool _hasHeader;

    [ObservableProperty]
    private string _selectedMethod = "auto";

    [ObservableProperty]
    private string? _inferenceError;

    [ObservableProperty]
    private bool _hasResult;

    [ObservableProperty]
    private string _resultInfo = string.Empty;

    public ObservableCollection<PythonOutputItem> Outputs { get; } = new();

    // ----- conversion -----

    [ObservableProperty]
    private string _outputPath;

    [ObservableProperty]
    private string _opsetText = "18";

    [ObservableProperty]
    private bool _disableZipMap = true;

    [ObservableProperty]
    private bool _validateConversion = true;

    [ObservableProperty]
    private bool _loadAfterConversion = true;

    [ObservableProperty]
    private string _inputTypesText = string.Empty;

    [ObservableProperty]
    private string? _conversionError;

    [ObservableProperty]
    private bool _hasConversionResult;

    [ObservableProperty]
    private string _conversionSummary = string.Empty;

    public ObservableCollection<string> ConversionDetails { get; } = new();

    public PythonModelViewModel(
        MainWindowViewModel shell,
        IPythonModelService service,
        IPythonRuntimeService runtime,
        IFilePickerService picker,
        IToastService toast,
        Func<IModelLoadCoordinator?> coordinator,
        PythonModel model)
    {
        _shell = shell;
        _service = service;
        _runtime = runtime;
        _picker = picker;
        _toast = toast;
        _coordinator = coordinator;
        Model = model;
        Title = model.Name;
        _outputPath = Path.ChangeExtension(model.FilePath, ".onnx");

        _runtime.Changed += OnRuntimeChanged;
        _ = RefreshRuntimeAsync();
    }

    private void OnRuntimeChanged() => Avalonia.Threading.Dispatcher.UIThread.Post(() => _ = RefreshRuntimeAsync());

    [RelayCommand]
    public async Task RefreshRuntimeAsync()
    {
        var active = await _runtime.GetActiveAsync().ConfigureAwait(true);
        RuntimeReady = active is { CanRunInference: true };
        RuntimeStatus = PythonRuntimeViewModel.Describe(active);
    }

    [RelayCommand]
    private void OpenPythonSettings() => _shell.ShowSettings();

    [RelayCommand]
    private void Back() => _shell.ShowDashboard();

    [RelayCommand]
    private void CancelOperation() => _operation?.Cancel();

    // ----- load -----

    [RelayCommand]
    public async Task LoadAsync()
    {
        LoadError = null;
        await ExecuteAsync(async token =>
        {
            var result = await _service.InspectAsync(Model.FilePath, IsTrusted, token).ConfigureAwait(true);
            if (result.IsFailure)
            {
                LoadError = Describe(result.Error!);
                return;
            }

            Info = result.Value;
            ShowInfo(result.Value!);
        }, onError: message => LoadError = message).ConfigureAwait(true);
    }

    private void ShowInfo(PythonModelInfo info)
    {
        FeaturesText = info.IsTextModel
            ? "raw text (one text per row)"
            : info.FeatureNames is { Count: > 0 }
                ? $"{info.FeatureNames.Count}: {string.Join(", ", info.FeatureNames)}"
                : info.FeatureCount is { } count ? count.ToString(CultureInfo.InvariantCulture) : "unknown";
        ClassesText = info.Classes is { Count: > 0 } ? string.Join(", ", info.Classes) : "-";
        MethodsText = string.Join(", ", info.Methods);
        VersionText = info.TrainedWithSklearn == null
            ? $"scikit-learn {info.RuntimeSklearn}"
            : info.TrainedWithSklearn == info.RuntimeSklearn
                ? $"scikit-learn {info.RuntimeSklearn}"
                : $"trained with scikit-learn {info.TrainedWithSklearn}, loaded with {info.RuntimeSklearn}";

        Parameters.Clear();
        foreach (var (key, value) in info.Parameters) Parameters.Add($"{key} = {value}");
        Warnings.Clear();
        foreach (var warning in info.Warnings) Warnings.Add(warning);

        foreach (var name in new[] { nameof(SummaryText), nameof(FeaturesText), nameof(ClassesText), nameof(MethodsText), nameof(VersionText) })
            OnPropertyChanged(name);

        HasHeader = info.FeatureNames is { Count: > 0 };
        InputText = SampleInput(info);
        SelectedMethod = "auto";
    }

    private static string SampleInput(PythonModelInfo info)
    {
        if (info.IsTextModel) return "sample text";
        var zeros = (info.FeatureCount ?? info.FeatureNames?.Count ?? 0) is var n and > 0
            ? string.Join(", ", Enumerable.Repeat("0", n))
            : string.Empty;
        return info.FeatureNames is { Count: > 0 }
            ? string.Join(", ", info.FeatureNames) + Environment.NewLine + zeros
            : zeros;
    }

    // ----- inference -----

    [RelayCommand]
    public async Task RunAsync()
    {
        InferenceError = null;
        HasResult = false;
        Outputs.Clear();

        var parsed = PythonInputParser.Parse(InputText, HasHeader, Info?.IsTextModel == true);
        if (parsed.IsFailure)
        {
            InferenceError = parsed.Error;
            return;
        }

        var table = parsed.Value!;
        await ExecuteAsync(async token =>
        {
            var result = await _service.PredictAsync(new PythonPredictionRequest(
                Model.FilePath, IsTrusted, SelectedMethod, table.Columns, table.Rows), token).ConfigureAwait(true);
            if (result.IsFailure)
            {
                InferenceError = Describe(result.Error!);
                return;
            }

            var value = result.Value!;
            foreach (var output in value.Outputs) Outputs.Add(new PythonOutputItem(output, value.Classes));
            ResultInfo = $"{value.RowCount} row(s) in {value.ElapsedMs} ms" +
                         (value.Warnings.Count > 0 ? $" - {value.Warnings.Count} warning(s): {value.Warnings[0]}" : string.Empty);
            HasResult = true;
        }, onError: message => InferenceError = message).ConfigureAwait(true);
    }

    // ----- conversion -----

    [RelayCommand]
    private async Task BrowseOutputAsync()
    {
        var path = await _picker.PickOnnxSavePathAsync(
            Path.GetFileName(OutputPath), Path.GetDirectoryName(OutputPath)).ConfigureAwait(true);
        if (path != null) OutputPath = path;
    }

    [RelayCommand]
    public async Task ConvertAsync()
    {
        ConversionError = null;
        HasConversionResult = false;
        ConversionDetails.Clear();

        if (!int.TryParse(OpsetText.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var opset) || opset is < 1 or > 30)
        {
            ConversionError = "The target opset must be a number between 1 and 30 (ONNX Studio opens models up to opset 18).";
            return;
        }

        var inputTypes = InputTypesText
            .Split(new[] { ',', ';', '\n' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToArray();

        await ExecuteAsync(async token =>
        {
            var result = await _service.ConvertAsync(new PythonConversionRequest(
                Model.FilePath, OutputPath.Trim(), IsTrusted, opset, DisableZipMap, inputTypes, ValidateConversion), token).ConfigureAwait(true);
            if (result.IsFailure)
            {
                ConversionError = Describe(result.Error!);
                return;
            }

            var conversion = result.Value!;
            ConversionSummary = $"Saved {conversion.OutputPath} (opset {conversion.TargetOpset}, {conversion.Size / 1024.0:0.#} KB)";
            ConversionDetails.Add("Inputs: " + string.Join("; ", conversion.Inputs.Select(i => i.Display)));
            ConversionDetails.Add("Outputs: " + string.Join("; ", conversion.Outputs.Select(o => o.Display)));
            if (conversion.TargetOpset != conversion.RequestedOpset)
                ConversionDetails.Add($"Opset lowered from {conversion.RequestedOpset} to {conversion.TargetOpset} (maximum supported by the installed packages).");
            ConversionDetails.Add("Validation: " + conversion.Validation.Detail);
            foreach (var warning in conversion.Warnings.Take(3)) ConversionDetails.Add("Warning: " + warning);
            HasConversionResult = true;
            _toast.Show("Model converted to ONNX");

            if (LoadAfterConversion && _coordinator() is { } coordinator)
            {
                await coordinator.LoadAsync(conversion.OutputPath).ConfigureAwait(true);
            }
        }, onError: message => ConversionError = message).ConfigureAwait(true);
    }

    // ----- helpers -----

    private async Task ExecuteAsync(Func<CancellationToken, Task> work, Action<string> onError)
    {
        if (IsWorking) return;
        IsWorking = true;
        _operation = new CancellationTokenSource();
        try
        {
            await work(_operation.Token).ConfigureAwait(true);
        }
        catch (OperationCanceledException)
        {
            onError("Operation cancelled.");
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidOperationException)
        {
            onError("The operation failed: " + ex.Message);
        }
        finally
        {
            _operation.Dispose();
            _operation = null;
            IsWorking = false;
        }
    }

    private static string Describe(PythonError error) => error.Code switch
    {
        PythonErrorCode.PredictionFailed or PythonErrorCode.ConversionFailed or PythonErrorCode.LoadFailed or PythonErrorCode.WorkerFailed
            when error.TechnicalDetails.Length > 0 => error.Message + Environment.NewLine + LastLine(error.TechnicalDetails),
        _ => error.Message
    };

    private static string LastLine(string trace)
        => trace.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).LastOrDefault() ?? string.Empty;

    public void Dispose()
    {
        _runtime.Changed -= OnRuntimeChanged;
        _operation?.Cancel();
    }
}
