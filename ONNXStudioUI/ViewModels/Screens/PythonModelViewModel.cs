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
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Python;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// Import step of a joblib / pickle file: check the Python runtime, confirm that the file is trusted,
/// load it as a <see cref="SklearnModel"/> and hand it over to the same screens as any other model
/// (inspector, inference, API, sandbox). It also converts the model to an ONNX file.
/// </summary>
public partial class PythonModelViewModel : ViewModelBase, IDisposable
{
    private readonly MainWindowViewModel _shell;
    private readonly IPythonModelService _service;
    private readonly IPythonRuntimeService _runtime;
    private readonly IFilePickerService _picker;
    private readonly IToastService _toast;
    private readonly Func<IModelLoadCoordinator?> _coordinator;
    private CancellationTokenSource? _operation;
    private string? _convertedPath;

    public PythonModel Model { get; }

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
    [NotifyPropertyChangedFor(nameof(IsLoaded))]
    [NotifyPropertyChangedFor(nameof(CanConvert))]
    [NotifyPropertyChangedFor(nameof(LoadButtonText))]
    private SklearnModel? _loadedModel;

    [ObservableProperty]
    private string? _loadError;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanLoad))]
    [NotifyPropertyChangedFor(nameof(CanConvert))]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    private bool _isWorking;

    public bool IsLoaded => LoadedModel != null;
    public string LoadButtonText => IsLoaded ? "Reload model" : "Load model";
    public bool IsIdle => !IsWorking;
    public bool CanLoad => IsTrusted && !IsWorking;
    public bool CanConvert => IsLoaded && !IsWorking;

    public string SummaryText => LoadedModel?.Info.Summary ?? string.Empty;
    public string FeaturesText { get; private set; } = string.Empty;
    public string ClassesText { get; private set; } = string.Empty;
    public string MethodsText { get; private set; } = string.Empty;
    public string VersionText { get; private set; } = string.Empty;
    public ObservableCollection<string> Warnings { get; } = new();

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

    /// <summary>Loads the file in the Python worker, registers it like any model and opens its inspector.</summary>
    [RelayCommand]
    public async Task LoadAsync()
    {
        if (!CanLoad) return;
        LoadError = null;
        await ExecuteAsync(async token =>
        {
            var result = await _service.InspectAsync(Model.FilePath, IsTrusted, token).ConfigureAwait(true);
            if (result.IsFailure)
            {
                LoadError = Describe(result.Error!);
                return;
            }

            token.ThrowIfCancellationRequested();
            var model = SklearnModel.Create(Model, result.Value!);
            ShowInfo(model);
            if (_coordinator() is { } coordinator) await coordinator.RegisterAsync(model).ConfigureAwait(true);
            _shell.ShowInspector(model);
        }, onError: message => LoadError = message).ConfigureAwait(true);
    }

    private void ShowInfo(SklearnModel model)
    {
        var info = model.Info;
        LoadedModel = model;
        FeaturesText = info.IsTextModel
            ? "raw text (one text per row)"
            : info.FeatureNames is { Count: > 0 }
                ? $"{info.FeatureNames.Count}: {string.Join(", ", info.FeatureNames)}"
                : info.FeatureCount is { } count ? count.ToString(CultureInfo.InvariantCulture) : "unknown";
        ClassesText = info.Classes is { Count: > 0 } ? string.Join(", ", info.Classes) : "-";
        MethodsText = string.Join(", ", info.Methods);
        VersionText = info.TrainedWithSklearn == null || info.TrainedWithSklearn == info.RuntimeSklearn
            ? $"scikit-learn {info.RuntimeSklearn}"
            : $"trained with scikit-learn {info.TrainedWithSklearn}, loaded with {info.RuntimeSklearn}";

        Warnings.Clear();
        foreach (var warning in info.Warnings) Warnings.Add(warning);

        foreach (var name in new[] { nameof(SummaryText), nameof(FeaturesText), nameof(ClassesText), nameof(MethodsText), nameof(VersionText) })
            OnPropertyChanged(name);
    }

    /// <summary>Opens the loaded model in one of the shared screens (inspector, playground, api, sandbox).</summary>
    [RelayCommand]
    private void OpenModel(string screen)
    {
        if (LoadedModel is { } model) Show(model, screen);
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
        _convertedPath = null;

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
            _convertedPath = conversion.OutputPath;
            _toast.Show("Model converted to ONNX");

            if (LoadAfterConversion && _coordinator() is { } coordinator)
            {
                await coordinator.LoadAsync(conversion.OutputPath).ConfigureAwait(true);
                if (_shell.FindModel(conversion.OutputPath) is { } loaded) _shell.ShowInspector(loaded);
            }
        }, onError: message => ConversionError = message).ConfigureAwait(true);
    }

    /// <summary>Opens the converted ONNX model in one of its screens, loading it first if needed.</summary>
    [RelayCommand]
    private async Task OpenConvertedAsync(string screen)
    {
        if (_convertedPath == null) return;
        var model = _shell.FindModel(_convertedPath);
        if (model == null && _coordinator() is { } coordinator)
        {
            await coordinator.LoadAsync(_convertedPath).ConfigureAwait(true);
            model = _shell.FindModel(_convertedPath);
        }
        if (model == null)
        {
            ConversionError = "The converted model could not be opened.";
            return;
        }

        Show(model, screen);
    }

    private void Show(IModel model, string screen)
    {
        switch (screen)
        {
            case "playground": _shell.ShowPlayground(model); break;
            case "api": _shell.ShowApiConfig(model); break;
            case "sandbox": _shell.ShowApiSandbox(model); break;
            default: _shell.ShowInspector(model); break;
        }
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
