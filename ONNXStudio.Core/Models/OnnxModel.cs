namespace ONNXStudio.Core.Models;

/// <summary>
/// A parsed ONNX model held by the ModelRegistry. The ONNX Runtime session
/// itself is owned by the InferenceSessionManager (separate lifecycle, LRU).
/// </summary>
public sealed class OnnxModel : IModel
{
    private readonly Lazy<IReadOnlyList<StructureNode>> _structure;

    public string Id { get; }
    public string Name { get; set; }
    public string FilePath { get; }
    public long FileSize { get; }
    public DateTime LoadedAt { get; }

    // Metadata
    public string ProducerName { get; }
    public string Domain { get; }
    public long OpsetVersion { get; }
    public string ModelVersion { get; }
    public string DocString { get; }
    public long IrVersion { get; }

    /// <summary>The metadata_props of the ONNX file (producers such as ONNX Studio store extra information there).</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    // Graph
    public ComputationGraph Graph { get; }

    // Schema
    public IReadOnlyList<TensorSchema> Inputs { get; }
    public IReadOnlyList<TensorSchema> Outputs { get; }

    // Weights (lazy-loaded raw values, metadata only)
    public IReadOnlyList<InitializerInfo> Initializers { get; }

    public OnnxModel(
        string id,
        string filePath,
        long fileSize,
        string producerName,
        string domain,
        long opsetVersion,
        string modelVersion,
        string docString,
        long irVersion,
        ComputationGraph graph,
        IReadOnlyList<TensorSchema> inputs,
        IReadOnlyList<TensorSchema> outputs,
        IReadOnlyList<InitializerInfo>? initializers = null,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        Id = id;
        Name = Path.GetFileNameWithoutExtension(filePath);
        FilePath = filePath;
        FileSize = fileSize;
        LoadedAt = DateTime.UtcNow;
        ProducerName = producerName;
        Domain = domain;
        OpsetVersion = opsetVersion;
        ModelVersion = modelVersion;
        DocString = docString;
        IrVersion = irVersion;
        Graph = graph;
        Inputs = inputs;
        Outputs = outputs;
        Initializers = initializers ?? Array.Empty<InitializerInfo>();
        Metadata = metadata ?? new Dictionary<string, string>();
        _structure = new(() => OnnxStructure.Build(this));
    }

    /// <summary>
    /// Human-readable file size (e.g. "98.5 MB").
    /// </summary>
    public string FileSizeDisplay => Utilities.FileSizeFormatter.Format(FileSize);

    public string Format => "ONNX";
    public string Producer => ProducerName;
    public string Description => DocString;
    public IReadOnlyList<string> Badges => [$"opset {OpsetVersion}"];

    public IReadOnlyList<KeyValuePair<string, string>> Facts =>
    [
        new("Producer", ProducerName),
        new("Opset", OpsetVersion.ToString(System.Globalization.CultureInfo.InvariantCulture)),
        .. Metadata.TryGetValue("onnxstudio.sklearn.class", out var source)
            ? new[] { new KeyValuePair<string, string>("Converted from", "scikit-learn " + source) }
            : []
    ];

    public string WeightsLabel => "Initializers";
    public IReadOnlyList<StructureNode> Structure => _structure.Value;
}
