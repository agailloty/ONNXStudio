using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;

namespace ONNXStudio.Core.Python;

/// <summary>What the Python worker found out about a joblib/pickle model.</summary>
public sealed class PythonModelInfo
{
    public string ClassName { get; init; } = string.Empty;
    public string Module { get; init; } = string.Empty;
    public bool IsPipeline { get; init; }
    public IReadOnlyList<PythonPipelineStep> Steps { get; init; } = Array.Empty<PythonPipelineStep>();
    public string FinalEstimator { get; init; } = string.Empty;
    public bool IsClassifier { get; init; }

    /// <summary>The model consumes raw text (first step is a text vectorizer).</summary>
    public bool IsTextModel { get; init; }

    public int? FeatureCount { get; init; }
    public IReadOnlyList<string>? FeatureNames { get; init; }
    public IReadOnlyList<string>? Classes { get; init; }

    /// <summary>Inference methods the model exposes (predict, predict_proba, ...).</summary>
    public IReadOnlyList<string> Methods { get; init; } = Array.Empty<string>();

    public IReadOnlyDictionary<string, string> Parameters { get; init; } = new Dictionary<string, string>();
    public string? TrainedWithSklearn { get; init; }
    public string? RuntimeSklearn { get; init; }
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();

    /// <summary>Estimator tree (pipeline steps, transformers...) with hyper-parameters and learned attributes.</summary>
    public PythonComponent? Components { get; init; }

    /// <summary>Versions of the Python packages installed where the model was inspected (missing ones are omitted).</summary>
    public IReadOnlyDictionary<string, string> Packages { get; init; } = new Dictionary<string, string>();

    /// <summary>Key of the ONNX metadata_props entry written by ONNX Studio when it converts a scikit-learn model.</summary>
    public const string OnnxMetadataKey = "onnxstudio.sklearn.info";

    public string Summary => IsPipeline
        ? $"Pipeline({string.Join(" -> ", Steps.Select(s => s.ClassName))})"
        : ClassName;

    /// <summary>Reads the JSON stored in an ONNX file by the converter; null when it is not valid.</summary>
    public static PythonModelInfo? TryParse(string json)
    {
        try { return JsonNode.Parse(json) is JsonObject root ? FromJson(root) : null; }
        catch (Exception ex) when (ex is System.Text.Json.JsonException or InvalidOperationException or FormatException) { return null; }
    }

    internal static PythonModelInfo FromJson(JsonObject json) => new()
    {
        ClassName = json["className"]?.GetValue<string>() ?? string.Empty,
        Module = json["module"]?.GetValue<string>() ?? string.Empty,
        IsPipeline = json["isPipeline"]?.GetValue<bool>() ?? false,
        Steps = (json["steps"] as JsonArray)?
            .Select(s => new PythonPipelineStep(s?["name"]?.GetValue<string>() ?? "", s?["className"]?.GetValue<string>() ?? ""))
            .ToArray() ?? Array.Empty<PythonPipelineStep>(),
        FinalEstimator = json["finalEstimator"]?.GetValue<string>() ?? string.Empty,
        IsClassifier = json["isClassifier"]?.GetValue<bool>() ?? false,
        IsTextModel = json["isTextModel"]?.GetValue<bool>() ?? false,
        FeatureCount = json["nFeaturesIn"]?.GetValue<int>(),
        FeatureNames = Strings(json["featureNamesIn"]),
        Classes = Strings(json["classes"]),
        Methods = Strings(json["methods"]) ?? Array.Empty<string>(),
        Parameters = (json["parameters"] as JsonObject)?
            .ToDictionary(p => p.Key, p => p.Value?.GetValue<string>() ?? string.Empty) ?? new Dictionary<string, string>(),
        TrainedWithSklearn = json["trainedWithSklearn"]?.GetValue<string>(),
        RuntimeSklearn = json["runtimeSklearn"]?.GetValue<string>(),
        Warnings = Strings(json["warnings"]) ?? Array.Empty<string>(),
        Components = json["components"] is JsonObject components ? PythonComponent.FromJson(components) : null,
        Packages = (json["packages"] as JsonObject)?
            .Where(p => p.Value is JsonValue v && v.TryGetValue<string>(out _))
            .ToDictionary(p => p.Key, p => p.Value!.GetValue<string>()) ?? new Dictionary<string, string>()
    };

    internal static string[]? Strings(JsonNode? node)
        => (node as JsonArray)?.Select(n => n is JsonValue v && v.TryGetValue<string>(out var s) ? s : n?.ToJsonString() ?? "null").ToArray();
}

public sealed record PythonPipelineStep(string Name, string ClassName);

/// <summary>A learned attribute (coef_, classes_, mean_...) of a fitted estimator: a bounded preview of its values.</summary>
public sealed class PythonFittedAttribute
{
    public string Name { get; init; } = string.Empty;

    /// <summary>scalar, array, sparse, dict, list or object.</summary>
    public string Kind { get; init; } = "scalar";

    public string? DType { get; init; }
    public IReadOnlyList<int> Shape { get; init; } = Array.Empty<int>();

    /// <summary>Number of values in the attribute (the preview in <see cref="Values"/> may be shorter).</summary>
    public int Count { get; init; } = 1;

    public IReadOnlyList<string> Values { get; init; } = Array.Empty<string>();
    public string Summary { get; init; } = string.Empty;

    public bool HasValues => Values.Count > 0;

    internal static PythonFittedAttribute FromJson(JsonNode json) => new()
    {
        Name = json["name"]?.GetValue<string>() ?? string.Empty,
        Kind = json["kind"]?.GetValue<string>() ?? "scalar",
        DType = json["dtype"]?.GetValue<string>(),
        Shape = (json["shape"] as JsonArray)?.Select(d => d?.GetValue<int>() ?? 0).ToArray() ?? Array.Empty<int>(),
        Count = json["count"]?.GetValue<int>() ?? 1,
        Values = (json["values"] as JsonArray)?.Select(PythonPredictionOutput.Format).ToArray() ?? Array.Empty<string>(),
        Summary = json["summary"]?.GetValue<string>() ?? string.Empty
    };
}

/// <summary>One estimator of the scikit-learn object graph (pipeline, step, transformer, model...).</summary>
public sealed class PythonComponent
{
    public string Name { get; init; } = string.Empty;
    public string ClassName { get; init; } = string.Empty;
    public string Module { get; init; } = string.Empty;

    /// <summary>Pipeline, Classifier, Regressor, Transformer, ColumnTransformer, FeatureUnion, Estimator or Passthrough.</summary>
    public string Kind { get; init; } = "Estimator";

    public string Description { get; init; } = string.Empty;
    public IReadOnlyList<KeyValuePair<string, string>> Parameters { get; init; } = Array.Empty<KeyValuePair<string, string>>();
    public IReadOnlyList<PythonFittedAttribute> Fitted { get; init; } = Array.Empty<PythonFittedAttribute>();
    public IReadOnlyList<PythonComponent> Children { get; init; } = Array.Empty<PythonComponent>();

    internal static PythonComponent FromJson(JsonObject json) => new()
    {
        Name = json["name"]?.GetValue<string>() ?? string.Empty,
        ClassName = json["className"]?.GetValue<string>() ?? string.Empty,
        Module = json["module"]?.GetValue<string>() ?? string.Empty,
        Kind = json["kind"]?.GetValue<string>() ?? "Estimator",
        Description = json["description"]?.GetValue<string>() ?? string.Empty,
        Parameters = (json["parameters"] as JsonObject)?
            .Select(p => KeyValuePair.Create(p.Key, p.Value is JsonValue v && v.TryGetValue<string>(out var s) ? s : p.Value?.ToJsonString() ?? string.Empty))
            .ToArray() ?? Array.Empty<KeyValuePair<string, string>>(),
        Fitted = (json["fitted"] as JsonArray)?.Where(f => f != null).Select(f => PythonFittedAttribute.FromJson(f!)).ToArray()
                 ?? Array.Empty<PythonFittedAttribute>(),
        Children = (json["children"] as JsonArray)?.OfType<JsonObject>().Select(FromJson).ToArray() ?? Array.Empty<PythonComponent>()
    };
}

public sealed record PythonPredictionRequest(
    string ModelPath,
    bool TrustConfirmed,
    string Method,
    IReadOnlyList<string>? Columns,
    IReadOnlyList<IReadOnlyList<string>> Rows);

/// <summary>One array returned by a model method (predict, predict_proba...).</summary>
public sealed class PythonPredictionOutput
{
    public string Name { get; }
    public string DType { get; }
    public IReadOnlyList<int> Shape { get; }
    public bool Truncated { get; }

    /// <summary>One formatted entry per input row.</summary>
    public IReadOnlyList<string> Rows { get; }

    /// <summary>The values as sent by the worker (nested arrays), before formatting.</summary>
    internal JsonNode? Raw { get; }

    public PythonPredictionOutput(string name, string dtype, IReadOnlyList<int> shape, bool truncated, IReadOnlyList<string> rows, JsonNode? raw = null)
    {
        Raw = raw;
        Name = name;
        DType = dtype;
        Shape = shape;
        Truncated = truncated;
        Rows = rows;
    }

    internal static PythonPredictionOutput FromJson(JsonNode json)
    {
        var rows = json["values"] is JsonArray values
            ? values.Select(Format).ToArray()
            : new[] { Format(json["values"]) };
        return new PythonPredictionOutput(
            json["name"]?.GetValue<string>() ?? "",
            json["dtype"]?.GetValue<string>() ?? "",
            (json["shape"] as JsonArray)?.Select(d => d!.GetValue<int>()).ToArray() ?? Array.Empty<int>(),
            json["truncated"]?.GetValue<bool>() ?? false,
            rows,
            json["values"]);
    }

    /// <summary>Compact text of a value: numbers use at most 6 significant digits.</summary>
    internal static string Format(JsonNode? node)
    {
        var builder = new StringBuilder();
        Append(builder, node);
        return builder.ToString();
    }

    private static void Append(StringBuilder builder, JsonNode? node)
    {
        switch (node)
        {
            case null:
                builder.Append("NaN");
                break;
            case JsonArray array:
                builder.Append('[');
                for (var i = 0; i < array.Count; i++)
                {
                    if (i > 0) builder.Append(", ");
                    Append(builder, array[i]);
                }
                builder.Append(']');
                break;
            case JsonValue value when value.TryGetValue<long>(out var integer):
                builder.Append(integer.ToString(CultureInfo.InvariantCulture));
                break;
            case JsonValue value when value.TryGetValue<double>(out var number):
                builder.Append(number.ToString("G6", CultureInfo.InvariantCulture));
                break;
            case JsonValue value when value.TryGetValue<string>(out var text):
                builder.Append(text);
                break;
            case JsonValue value when value.TryGetValue<bool>(out var flag):
                builder.Append(flag ? "True" : "False");
                break;
            default:
                builder.Append(node.ToJsonString());
                break;
        }
    }
}

public sealed class PythonPredictionResult
{
    public IReadOnlyList<PythonPredictionOutput> Outputs { get; }
    public IReadOnlyList<string>? Classes { get; }
    public int RowCount { get; }
    public long ElapsedMs { get; }
    public IReadOnlyList<string> Warnings { get; }

    public PythonPredictionResult(IReadOnlyList<PythonPredictionOutput> outputs, IReadOnlyList<string>? classes, int rowCount,
        long elapsedMs, IReadOnlyList<string> warnings)
    {
        Outputs = outputs;
        Classes = classes;
        RowCount = rowCount;
        ElapsedMs = elapsedMs;
        Warnings = warnings;
    }
}

public sealed record PythonConversionRequest(
    string ModelPath,
    string OutputPath,
    bool TrustConfirmed,
    int TargetOpset = 18,
    bool DisableZipMap = true,
    IReadOnlyList<string>? InputTypes = null,
    bool Validate = true);

public sealed record PythonOnnxTensor(string Name, int ElementType, IReadOnlyList<long?> Shape)
{
    public string Display => $"{Name} [{string.Join(", ", Shape.Select(d => d?.ToString() ?? "?"))}]";
}

public sealed record PythonConversionValidation(bool Performed, bool? Passed, string Detail);

public sealed class PythonConversionResult
{
    public string OutputPath { get; init; } = string.Empty;
    public int TargetOpset { get; init; }
    public int RequestedOpset { get; init; }
    public long Size { get; init; }
    public IReadOnlyList<PythonOnnxTensor> Inputs { get; init; } = Array.Empty<PythonOnnxTensor>();
    public IReadOnlyList<PythonOnnxTensor> Outputs { get; init; } = Array.Empty<PythonOnnxTensor>();
    public PythonConversionValidation Validation { get; init; } = new(false, null, string.Empty);
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();

    internal static PythonConversionResult FromJson(JsonObject json) => new()
    {
        OutputPath = json["outputPath"]?.GetValue<string>() ?? string.Empty,
        TargetOpset = json["targetOpset"]?.GetValue<int>() ?? 0,
        RequestedOpset = json["requestedOpset"]?.GetValue<int>() ?? 0,
        Size = json["size"]?.GetValue<long>() ?? 0,
        Inputs = Tensors(json["inputs"]),
        Outputs = Tensors(json["outputs"]),
        Validation = json["validation"] is JsonObject v
            ? new PythonConversionValidation(
                v["performed"]?.GetValue<bool>() ?? false,
                v["passed"]?.GetValue<bool>(),
                v["detail"]?.GetValue<string>() ?? string.Empty)
            : new PythonConversionValidation(false, null, string.Empty),
        Warnings = PythonModelInfo.Strings(json["warnings"]) ?? Array.Empty<string>()
    };

    private static PythonOnnxTensor[] Tensors(JsonNode? node)
        => (node as JsonArray)?.Select(t => new PythonOnnxTensor(
            t?["name"]?.GetValue<string>() ?? "",
            t?["elementType"]?.GetValue<int>() ?? 0,
            (t?["shape"] as JsonArray)?.Select(d => d?.GetValue<long>()).ToArray() ?? Array.Empty<long?>())).ToArray()
           ?? Array.Empty<PythonOnnxTensor>();
}

/// <summary>A joblib / pickle file opened in the studio.</summary>
public sealed class PythonModel
{
    public static readonly IReadOnlyList<string> SupportedExtensions = new[] { ".joblib", ".jbl", ".pkl", ".pickle", ".pck", ".sav" };

    public string Id { get; }
    public string Name { get; }
    public string FilePath { get; }
    public long FileSize { get; }
    public DateTime LoadedAt { get; }

    public PythonModel(string id, string filePath, long fileSize)
    {
        Id = id;
        FilePath = filePath;
        Name = Path.GetFileName(filePath);
        FileSize = fileSize;
        LoadedAt = DateTime.UtcNow;
    }

    public string FileSizeDisplay => Utilities.FileSizeFormatter.Format(FileSize);

    public static bool IsPythonModelFile(string path)
        => SupportedExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase);
}
