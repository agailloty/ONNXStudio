namespace ONNXStudio.Core.Models;

/// <summary>
/// A parsed ONNX model held by the ModelRegistry. The ONNX Runtime session
/// itself is owned by the InferenceSessionManager (separate lifecycle, LRU).
/// </summary>
public sealed class OnnxModel
{
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

    // Graph
    public ComputationGraph Graph { get; }

    // Schema
    public IReadOnlyList<TensorSchema> Inputs { get; }
    public IReadOnlyList<TensorSchema> Outputs { get; }

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
        IReadOnlyList<TensorSchema> outputs)
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
    }

    /// <summary>
    /// Human-readable file size (e.g. "98.5 MB").
    /// </summary>
    public string FileSizeDisplay => Utilities.FileSizeFormatter.Format(FileSize);
}
