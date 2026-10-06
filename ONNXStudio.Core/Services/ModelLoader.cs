using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Utilities;
using Microsoft.ML.OnnxRuntime;

namespace ONNXStudio.Core.Services;

/// <summary>
/// Loads and validates ONNX model files (US-001).
/// </summary>
public interface IModelLoader
{
    Task<Result<OnnxModel, ModelLoadError>> LoadAsync(string filePath, CancellationToken cancellationToken = default);
}

public sealed class ModelLoader : IModelLoader
{
    // Per US-001: opsets 1..18 are supported.
    private const long MaxSupportedOpset = 18;

    private readonly OnnxStudioOptions _options;
    private readonly ILogger<ModelLoader> _logger;

    public ModelLoader(IOptions<OnnxStudioOptions> options, ILogger<ModelLoader> logger)
    {
        _options = options.Value;
        _logger = logger;
    }

    public async Task<Result<OnnxModel, ModelLoadError>> LoadAsync(string filePath, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            return Result<OnnxModel, ModelLoadError>.Failure(
                new ModelLoadError(ModelLoadErrorCode.FileNotFound, "No file path was provided."));
        }

        if (!File.Exists(filePath))
        {
            return Result<OnnxModel, ModelLoadError>.Failure(
                new ModelLoadError(ModelLoadErrorCode.FileNotFound, $"The file '{Path.GetFileName(filePath)}' does not exist.",
                    $"Path: {filePath}"));
        }

        if (!string.Equals(Path.GetExtension(filePath), ".onnx", StringComparison.OrdinalIgnoreCase))
        {
            return Result<OnnxModel, ModelLoadError>.Failure(
                new ModelLoadError(ModelLoadErrorCode.InvalidFileExtension,
                    $"'{Path.GetFileName(filePath)}' is not an ONNX model. Please select a .onnx file."));
        }

        var fileInfo = new FileInfo(filePath);
        var maxSizeBytes = (long)_options.MaxModelSizeMB * 1024 * 1024;
        if (fileInfo.Length > maxSizeBytes)
        {
            return Result<OnnxModel, ModelLoadError>.Failure(
                new ModelLoadError(ModelLoadErrorCode.FileTooLarge,
                    $"The model file is too large ({FileSizeFormatter.Format(fileInfo.Length)}). " +
                    $"The maximum supported size is {_options.MaxModelSizeMB} MB."));
        }

        byte[] bytes;
        try
        {
            bytes = await File.ReadAllBytesAsync(filePath, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Could not read model file {FilePath}", filePath);
            return Result<OnnxModel, ModelLoadError>.Failure(
                new ModelLoadError(ModelLoadErrorCode.LoadFailed, "The model file could not be read.",
                    ex.Message, ex));
        }

        // ONNX files are protobuf ModelProto messages; they normally start
        // with field 1 (ir_version) => 0x08. Anything else is not an ONNX model.
        if (!OnnxProtoParser.LooksLikeOnnxModel(bytes))
        {
            return Result<OnnxModel, ModelLoadError>.Failure(
                new ModelLoadError(ModelLoadErrorCode.InvalidFileFormat,
                    "This file is not a valid ONNX model (unexpected file signature).",
                    $"First bytes: {BitConverter.ToString(bytes, 0, Math.Min(8, bytes.Length))}"));
        }

        // Parse metadata / graph with the built-in protobuf parser.
        OnnxProtoParser.RawModel raw;
        try
        {
            raw = OnnxProtoParser.Parse(bytes);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to parse ONNX protobuf of {FilePath}", filePath);
            return Result<OnnxModel, ModelLoadError>.Failure(
                new ModelLoadError(ModelLoadErrorCode.InvalidFileFormat,
                    "This ONNX file appears to be corrupted.", ex.Message, ex));
        }

        if (raw.OpsetVersion > MaxSupportedOpset)
        {
            return Result<OnnxModel, ModelLoadError>.Failure(
                new ModelLoadError(ModelLoadErrorCode.UnsupportedOpsetVersion,
                    $"This model requires ONNX opset {raw.OpsetVersion}. " +
                    $"Please export your model with opset <= {MaxSupportedOpset}.",
                    $"Opset: {raw.OpsetVersion}, IR version: {raw.IrVersion}"));
        }

        // Validate the model with ONNX Runtime (it also owns the inference session later).
        try
        {
            using var session = new InferenceSession(bytes);
            if (raw.Inputs.Count == 0)
            {
                return Result<OnnxModel, ModelLoadError>.Failure(
                    new ModelLoadError(ModelLoadErrorCode.InvalidModel,
                        "This model has no inputs and cannot be used for inference."));
            }
        }
        catch (OnnxRuntimeException ex)
        {
            _logger.LogError(ex, "ONNX Runtime rejected model {FilePath}", filePath);
            return Result<OnnxModel, ModelLoadError>.Failure(
                new ModelLoadError(ModelLoadErrorCode.InvalidModel,
                    "This ONNX model could not be loaded: " + ShortenOrtMessage(ex.Message), ex.Message, ex));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error loading {FilePath}", filePath);
            return Result<OnnxModel, ModelLoadError>.Failure(
                new ModelLoadError(ModelLoadErrorCode.LoadFailed,
                    "An unexpected error occurred while loading the model.", ex.Message, ex));
        }

        var model = BuildModel(filePath, fileInfo.Length, raw);        _logger.LogInformation("Loaded ONNX model {ModelName} (opset {Opset}, {Nodes} nodes, {Inputs} inputs, {Outputs} outputs)",
            model.Name, model.OpsetVersion, model.Graph.Nodes.Count, model.Inputs.Count, model.Outputs.Count);

        return Result<OnnxModel, ModelLoadError>.Success(model);
    }

    private static OnnxModel BuildModel(string filePath, long fileSize, OnnxProtoParser.RawModel raw)
    {
        var nodes = raw.Nodes.Select((n, i) => new GraphNode(
            id: $"node_{i}",
            name: string.IsNullOrEmpty(n.Name) ? $"{n.OpType}_{i}" : n.Name,
            opType: n.OpType,
            domain: n.Domain,
            attributes: n.Attributes,
            inputIds: n.Inputs,
            outputIds: n.Outputs)).ToList();

        var edges = BuildEdges(raw);

        var inputs = raw.Inputs
            .Where(v => !string.IsNullOrEmpty(v.Name))
            .Select(v => new TensorSchema(
                v.Name,
                DataTypeExtensions.FromOnnxRuntimeElementType(v.ElementType),
                v.Shape))
            .ToList();

        var outputs = raw.Outputs
            .Where(v => !string.IsNullOrEmpty(v.Name))
            .Select(v => new TensorSchema(
                v.Name,
                DataTypeExtensions.FromOnnxRuntimeElementType(v.ElementType),
                v.Shape))
            .ToList();

        var initializers = raw.Initializers
            .Select(t => new InitializerInfo(t.Name, t.Dims, t.Dims.Aggregate(1L, (a, b) => a * b)))
            .ToList();

        return new OnnxModel(
            id: Guid.NewGuid().ToString("N"),
            filePath: filePath,
            fileSize: fileSize,
            producerName: string.IsNullOrEmpty(raw.ProducerName) ? "Unknown" : raw.ProducerName,
            domain: raw.Domain,
            opsetVersion: raw.OpsetVersion,
            modelVersion: raw.ModelVersion == 0 ? "1" : raw.ModelVersion.ToString(),
            docString: raw.DocString,
            irVersion: raw.IrVersion,
            graph: new ComputationGraph(nodes, edges),
            inputs: inputs,
            outputs: outputs,
            initializers: initializers);
    }

    /// <summary>
    /// Builds tensor-flow edges by mapping each consumed tensor to its producer node.
    /// </summary>
    private static List<GraphEdge> BuildEdges(OnnxProtoParser.RawModel raw)
    {
        var producers = new Dictionary<string, string>(); // tensor name -> node id
        for (int i = 0; i < raw.Nodes.Count; i++)
        {
            foreach (var output in raw.Nodes[i].Outputs)
            {
                producers[output] = $"node_{i}";
            }
        }

        var edges = new List<GraphEdge>();
        var seen = new HashSet<(string, string, string)>();
        for (int i = 0; i < raw.Nodes.Count; i++)
        {
            var consumer = $"node_{i}";
            foreach (var input in raw.Nodes[i].Inputs)
            {
                if (producers.TryGetValue(input, out var producer) && producer != consumer)
                {
                    var key = (input, producer, consumer);
                    if (seen.Add(key))
                    {
                        edges.Add(new GraphEdge(input, producer, consumer));
                    }
                }
            }
        }
        return edges;
    }

    private static string ShortenOrtMessage(string message)
    {
        var newline = message.IndexOf('\n');
        var first = newline > 0 ? message[..newline] : message;
        return first.Length > 200 ? first[..200] + "..." : first;
    }
}
