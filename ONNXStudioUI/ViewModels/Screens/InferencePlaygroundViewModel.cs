using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// One editable input field of the playground form.
/// </summary>
public partial class PlaygroundFieldViewModel : ObservableObject
{
    private readonly TensorSchema _schema;
    private readonly IFilePickerService _filePicker;

    public FormField Field { get; }
    public string Label => Field.Label;
    public FormFieldKind Kind => Field.Kind;
    public string TypeDisplay => Field.TypeDisplay;
    public string Description => Field.Description;
    public bool IsNumber => Kind == FormFieldKind.Number;
    public bool IsVector => Kind == FormFieldKind.Vector;
    public bool IsImage => Kind == FormFieldKind.Image;
    public bool IsText => Kind == FormFieldKind.Text;

    [ObservableProperty]
    private string _valueText = string.Empty;

    [ObservableProperty]
    private string? _imagePath;

    [ObservableProperty]
    private string? _error;

    /// <summary>Decoded image tensor (CHW, normalized 0..1), set after upload.</summary>
    private float[]? _imageTensor;

    public PlaygroundFieldViewModel(FormField field, TensorSchema schema, IFilePickerService filePicker)
    {
        Field = field;
        _schema = schema;
        _filePicker = filePicker;

        // Sensible defaults
        ValueText = Kind switch
        {
            FormFieldKind.Number => field.DefaultValue.ToString(CultureInfo.InvariantCulture),
            FormFieldKind.Vector => DefaultVector(schema),
            _ => string.Empty
        };
    }

    private static string DefaultVector(TensorSchema schema)
    {
        var staticDims = schema.Shape.Where(d => d.HasValue).Select(d => d!.Value).ToList();
        var count = staticDims.Count > 0 ? (int)staticDims.Aggregate(1L, (a, b) => a * b) : 1;
        if (count > 16)
        {
            return string.Empty;
        }
        return string.Join(", ", Enumerable.Repeat("0", count));
    }

    [RelayCommand]
    private async Task UploadImageAsync()
    {
        var path = await _filePicker.PickImageFileAsync().ConfigureAwait(true);
        if (path == null)
        {
            return;
        }

        try
        {
            _imageTensor = ImageTensorDecoder.DecodeToChw(
                path, Field.ExpectedChannels, Field.ExpectedWidth, Field.ExpectedHeight);
            ImagePath = path;
            Error = null;
        }
        catch (Exception ex)
        {
            Error = "Could not decode image: " + ex.Message;
        }
    }

    /// <summary>
    /// Builds the inference input value from the user input.
    /// </summary>
    public Result<InferenceInputValue, string> ToInputValue()
    {
        switch (Kind)
        {
            case FormFieldKind.Number:
                if (!double.TryParse(ValueText, NumberStyles.Float, CultureInfo.InvariantCulture, out var number))
                {
                    return Result<InferenceInputValue, string>.Failure("enter a valid number");
                }
                return Result<InferenceInputValue, string>.Success(
                    _schema.Shape.Count == 0 || _schema.Shape.All(d => d is 1 or null)
                        ? InferenceInputValue.Scalar(number)
                        : InferenceInputValue.Scalars(number));

            case FormFieldKind.Vector:
                var parts = ValueText.Split(new[] { ',', ' ', ';' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 0)
                {
                    return Result<InferenceInputValue, string>.Failure("enter comma-separated values");
                }
                var values = new double[parts.Length];
                for (int i = 0; i < parts.Length; i++)
                {
                    if (!double.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out values[i]))
                    {
                        return Result<InferenceInputValue, string>.Failure($"'{parts[i]}' is not a number");
                    }
                }

                var shape = ResolveVectorShape(_schema, values.Length);
                return Result<InferenceInputValue, string>.Success(
                    InferenceInputValue.Array(values, shape));

            case FormFieldKind.Image:
                if (_imageTensor == null)
                {
                    return Result<InferenceInputValue, string>.Failure("upload an image first");
                }
                return Result<InferenceInputValue, string>.Success(
                    InferenceInputValue.Tensor(_imageTensor,
                        new[] { 1L, Field.ExpectedChannels, Field.ExpectedHeight, Field.ExpectedWidth }));

            default:
                return Result<InferenceInputValue, string>.Failure("text inputs are not supported yet");
        }
    }

    private static long[] ResolveVectorShape(TensorSchema schema, int length)
    {
        if (schema.Shape.Count == 1)
        {
            return new[] { (long)length };
        }

        var shape = new long[schema.Shape.Count];
        for (int i = 0; i < schema.Shape.Count; i++)
        {
            shape[i] = schema.Shape[i] ?? 0;
        }
        if (schema.Shape.Count == 2 && schema.Shape[1].HasValue && length == schema.Shape[1])
        {
            shape[0] = 1;
        }
        else if (shape.Any(d => d == 0))
        {
            // dynamic: wrap as a single row
            return new[] { 1L, length };
        }
        return shape;
    }
}

/// <summary>
/// One output tensor rendered in the results panel.
/// </summary>
public partial class OutputItemViewModel : ObservableObject
{
    public string Name { get; }
    public string ShapeDisplay { get; }
    public string Kind { get; }
    public string ScalarDisplay { get; }
    public bool HasScalar => !string.IsNullOrEmpty(ScalarDisplay);
    public ObservableCollection<double> TopValues { get; } = new();
    public double MaxTop => TopValues.Count > 0 ? TopValues.Max() : 1.0;

    public OutputItemViewModel(TensorOutput output)
    {
        Name = output.Name;
        ShapeDisplay = "[" + string.Join(", ", output.Shape) + "]";
        ScalarDisplay = output.Data.Length == 1
            ? System.Convert.ToDouble(output.Data.GetValue(0)!).ToString("0.####")
            : string.Empty;

        var values = new List<double>();
        foreach (var v in output.Data)
        {
            values.Add(System.Convert.ToDouble(v));
        }

        if (output.Shape.Count == 2 && output.Shape[0] == 1 && values.Count > 4)
        {
            // Classification-like: top 5 classes
            Kind = "Top classes";
            foreach (var value in values.OrderByDescending(v => v).Take(5))
            {
                TopValues.Add(value);
            }
        }
        else if (values.Count <= 16)
        {
            Kind = "Values";
            foreach (var value in values)
            {
                TopValues.Add(value);
            }
        }
        else
        {
            Kind = "First values";
            foreach (var value in values.Take(16))
            {
                TopValues.Add(value);
            }
        }
    }
}

/// <summary>
/// Inference playground (S-05 / US-003, US-004): dynamic form generated from
/// the model schema, real inference execution and result display.
/// </summary>
public partial class InferencePlaygroundViewModel : ViewModelBase
{
    private readonly MainWindowViewModel _shell;
    private readonly IInferenceService _inference;
    private readonly OnnxModel _model;

    [ObservableProperty]
    private ObservableCollection<PlaygroundFieldViewModel> _fields = new();

    [ObservableProperty]
    private ObservableCollection<OutputItemViewModel> _outputs = new();

    [ObservableProperty]
    private bool _isRunning;

    [ObservableProperty]
    private bool _hasResult;

    [ObservableProperty]
    private long _executionTimeMs;

    [ObservableProperty]
    private string? _error;

    public OnnxModel Model => _model;

    public InferencePlaygroundViewModel(
        MainWindowViewModel shell,
        IInferenceService inference,
        IFormGenerationService formGeneration,
        IFilePickerService filePicker,
        OnnxModel model)
    {
        _shell = shell;
        _inference = inference;
        _model = model;
        Title = model.Name + " - Inference";

        var schemaByName = model.Inputs.ToDictionary(i => i.Name);
        foreach (var field in formGeneration.GenerateFields(model))
        {
            Fields.Add(new PlaygroundFieldViewModel(field, schemaByName[field.InputName], filePicker));
        }
    }

    [RelayCommand]
    private void Back()
    {
        _shell.ShowDashboard();
    }

    [RelayCommand]
    private void SwitchToInspector()
    {
        _shell.ShowInspector(_model);
    }

    [RelayCommand]
    private void SwitchToApi()
    {
        _shell.ShowApiConfig(_model);
    }

    [RelayCommand]
    private async Task RunInferenceAsync()
    {
        Error = null;

        // Parse and validate every field
        var inputs = new Dictionary<string, InferenceInputValue>();
        foreach (var field in Fields)
        {
            var parsed = field.ToInputValue();
            if (parsed.IsFailure)
            {
                field.Error = parsed.Error;
                Error = $"Input '{field.Label}': {parsed.Error}";
                return;
            }
            field.Error = null;
            inputs[field.Field.InputName] = parsed.Value!;
        }

        IsRunning = true;
        try
        {
            var result = await _inference.RunAsync(_model, inputs).ConfigureAwait(true);
            if (result.IsFailure)
            {
                Error = result.Error!.Message;
                return;
            }

            Outputs = new ObservableCollection<OutputItemViewModel>(
                result.Value!.Outputs.Select(o => new OutputItemViewModel(o)));
            ExecutionTimeMs = result.Value.ExecutionTimeMs;
            HasResult = true;
            _shell.ShowToast($"Inference completed in {ExecutionTimeMs} ms");
        }
        finally
        {
            IsRunning = false;
        }
    }
}
