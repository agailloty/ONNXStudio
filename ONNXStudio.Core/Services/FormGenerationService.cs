using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Services;

public enum FormFieldKind
{
    Number,   // single numeric value (slider / numeric input)
    Vector,   // comma-separated numbers, one per tensor element
    Image,    // 4D float tensor [1, C, H, W]
    Text      // string tensor
}

/// <summary>
/// UI-agnostic description of an input field generated from the model
/// schema (US-003). The UI layer renders whatever control fits the Kind.
/// </summary>
public sealed class FormField
{
    public string InputName { get; }
    public string Label { get; }
    public FormFieldKind Kind { get; }
    public string TypeDisplay { get; }
    public string ShapeDisplay { get; }
    public bool IsRequired { get; } = true;
    public double DefaultValue { get; set; }
    public double? Min { get; set; }
    public double? Max { get; set; }
    public double Step { get; set; } = 1.0;
    public string Description { get; set; } = string.Empty;

    // Image-specific
    public int ExpectedChannels { get; set; }
    public int ExpectedWidth { get; set; }
    public int ExpectedHeight { get; set; }

    public FormField(string inputName, string label, FormFieldKind kind, string typeDisplay, string shapeDisplay)
    {
        InputName = inputName;
        Label = label;
        Kind = kind;
        TypeDisplay = typeDisplay;
        ShapeDisplay = shapeDisplay;
    }
}

/// <summary>
/// Generates inference form fields from the model input schemas (US-003).
/// </summary>
public interface IFormGenerationService
{
    IReadOnlyList<FormField> GenerateFields(OnnxModel model);
}

public sealed class FormGenerationService : IFormGenerationService
{
    public IReadOnlyList<FormField> GenerateFields(OnnxModel model)
    {
        var fields = new List<FormField>();
        foreach (var input in model.Inputs)
        {
            fields.Add(GenerateField(input));
        }
        return fields;
    }

    private static FormField GenerateField(TensorSchema input)
    {
        var rank = input.Shape.Count;
        var staticDims = input.Shape.Where(d => d.HasValue).Select(d => d!.Value).ToList();
        var elementCount = staticDims.Count > 0 ? staticDims.Aggregate(1L, (a, b) => a * b) : 1;

        // String tensors -> text input
        if (input.Type == DataType.String)
        {
            return new FormField(input.Name, input.Name, FormFieldKind.Text,
                input.ToDisplayString(), input.ToDisplayString())
            {
                Description = string.IsNullOrEmpty(input.Description)
                    ? "Enter text"
                    : input.Description
            };
        }

        // Rank 4 image tensor [1, C, H, W] -> image input
        if (rank == 4 && input.Type == DataType.Float32 && input.Shape[0] is 1 or null)
        {
            var channels = (int)(input.Shape[1] ?? 3);
            var height = (int)(input.Shape[2] ?? 224);
            var width = (int)(input.Shape[3] ?? 224);
            return new FormField(input.Name, input.Name, FormFieldKind.Image,
                input.ToDisplayString(), input.ToDisplayString())
            {
                ExpectedChannels = channels,
                ExpectedHeight = height,
                ExpectedWidth = width,
                Description = $"Expected image: {width}x{height}x{channels}"
            };
        }

        // Rank 0 or 1 element -> number input
        if (rank == 0 || elementCount == 1)
        {
            return new FormField(input.Name, input.Name, FormFieldKind.Number,
                input.ToDisplayString(), input.ToDisplayString())
            {
                DefaultValue = 0.0,
                Description = "Single value"
            };
        }

        // Everything else -> vector input (comma-separated values)
        return new FormField(input.Name, input.Name, FormFieldKind.Vector,
            input.ToDisplayString(), input.ToDisplayString())
        {
            Description = $"Enter {elementCount} comma-separated values"
        };
    }
}
