using CommunityToolkit.Mvvm.ComponentModel;

namespace ONNXStudio.Mocks.Models;

/// <summary>
/// Maps one field of the incoming JSON payload to a model input (optionally to a
/// column of a multi-column input tensor, e.g. float_input[0] = MedInc).
/// </summary>
public partial class ApiFieldMapping : ObservableObject
{
    /// <summary>Model input tensor the field feeds.</summary>
    public string SourceName { get; set; } = string.Empty;

    /// <summary>ONNX element type of the source tensor (float32, int64, ...).</summary>
    public string SourceType { get; set; } = "float32";

    /// <summary>Column index inside the source tensor, when the tensor is 2D.</summary>
    public int? SourceColumn { get; set; }

    public string SourceDisplay => SourceColumn is int c
        ? SourceName + "[" + c + "] - " + SourceType
        : SourceName + " - " + SourceType;

    [ObservableProperty]
    private string _jsonName = string.Empty;

    [ObservableProperty]
    private string _jsonType = "number"; // number / array / string / boolean

    [ObservableProperty]
    private bool _isRequired = true;

    [ObservableProperty]
    private string _defaultValue = string.Empty;

    [ObservableProperty]
    private string _sampleValue = string.Empty;

    [ObservableProperty]
    private string _description = string.Empty;
}
