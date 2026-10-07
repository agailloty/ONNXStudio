using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
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
    [ObservableProperty] private string _shapeText = string.Empty;
    public bool HasDynamicShape => _schema.HasDynamicDimension;

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
        long count = 1;
        foreach (var dimension in schema.Shape)
        {
            if (dimension is > 16 or <= 0 || count > 16 / (dimension ?? 1)) return string.Empty;
            count *= dimension ?? 1;
        }
        return string.Join(", ", Enumerable.Repeat("0", (int)count));
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
            _imageTensor = null;
            ImagePath = null;
            Error = "Could not decode image: " + ex.Message;
        }
    }

    /// <summary>
    /// Builds the inference input value from the user input.
    /// </summary>
    public Result<InferenceInputValue, string> ToInputValue()
    {
        if (Kind != FormFieldKind.Image)
            return TensorInputParser.Parse(_schema, ValueText, ShapeText);
        if (_imageTensor == null)
            return Result<InferenceInputValue, string>.Failure("Upload an image first.");
        return Result<InferenceInputValue, string>.Success(InferenceInputValue.Tensor(_imageTensor,
            new[] { 1L, Field.ExpectedChannels, Field.ExpectedHeight, Field.ExpectedWidth }));
    }

    public void Reset()
    {
        ValueText = Kind == FormFieldKind.Number ? "0" : Kind == FormFieldKind.Vector ? DefaultVector(_schema) : string.Empty;
        ShapeText = string.Empty;
        ImagePath = null;
        _imageTensor = null;
        Error = null;
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
    public ObservableCollection<string> DisplayValues { get; } = new();
    public ObservableCollection<double> TopValues { get; } = new();
    public double MaxTop => TopValues.Count > 0 ? TopValues.Max() : 1.0;

    public OutputItemViewModel(TensorOutput output)
    {
        Name = output.Name;
        ShapeDisplay = "[" + string.Join(", ", output.Shape) + "]";
        if (output.Type == DataType.String)
        {
            Kind = "Text";
            ScalarDisplay = string.Join("\n", output.Data.Cast<string>().Take(16));
            return;
        }
        ScalarDisplay = output.Data.Length == 1
            ? System.Convert.ToString(output.Data.GetValue(0), CultureInfo.InvariantCulture) ?? string.Empty
            : string.Empty;

        var values = new List<double>();
        foreach (var v in output.Data)
        {
            values.Add(System.Convert.ToDouble(v));
        }

        if (output.Shape.Count == 2 && output.Shape[0] == 1 && values.Count > 4 && output.Type is DataType.Float32 or DataType.Float64 or DataType.Float16)
        {
            // Classification-like: top 5 classes
            Kind = "Top classes (raw scores)";
            foreach (var item in values.Select((value, index) => (value, index)).OrderByDescending(x => x.value).Take(5))
                DisplayValues.Add($"Class {item.index}: {item.value:G6}");
            foreach (var value in values.OrderByDescending(v => v).Take(5))
            {
                TopValues.Add(value);
            }
        }
        else if (values.Count <= 16)
        {
            Kind = "Values";
            foreach (var value in output.Data) DisplayValues.Add(System.Convert.ToString(value, CultureInfo.InvariantCulture) ?? "");
            foreach (var value in values)
            {
                TopValues.Add(value);
            }
        }
        else
        {
            Kind = "First values";
            foreach (var value in output.Data.Cast<object>().Take(16)) DisplayValues.Add(System.Convert.ToString(value, CultureInfo.InvariantCulture) ?? "");
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
    private readonly IModel _model;

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

    public IModel Model => _model;

    public InferencePlaygroundViewModel(
        MainWindowViewModel shell,
        IInferenceService inference,
        IFormGenerationService formGeneration,
        IFilePickerService filePicker,
        IModel model)
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
    private void Reset()
    {
        foreach (var field in Fields) field.Reset();
        Outputs.Clear();
        HasResult = false;
        Error = null;
    }

    [RelayCommand]
    private async Task RunInferenceAsync()
    {
        Error = null;
        HasResult = false;
        Outputs.Clear();

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
            var result = await Task.Run(() => _inference.RunAsync(_model, inputs));
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
        catch (Exception)
        {
            Error = "Inference could not complete. Check the inputs and try again.";
        }
        finally
        {
            IsRunning = false;
        }
    }
}
