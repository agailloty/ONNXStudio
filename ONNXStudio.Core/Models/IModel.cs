namespace ONNXStudio.Core.Models;

/// <summary>
/// A model opened in the studio, whatever its technology (ONNX, scikit-learn, later ML.NET...).
/// The inspector, inference, API and sandbox screens only depend on this contract:
/// a computation graph, input/output schemas, a structure tree and an
/// <see cref="Services.IInferenceBackend"/> able to run it.
/// </summary>
public interface IModel
{
    string Id { get; }
    string Name { get; }
    string FilePath { get; }
    long FileSize { get; }
    string FileSizeDisplay { get; }
    DateTime LoadedAt { get; }

    /// <summary>Technology of the model: "ONNX", "scikit-learn"...</summary>
    string Format { get; }

    /// <summary>Who produced the model (exporter, library and version...).</summary>
    string Producer { get; }

    string Description { get; }

    /// <summary>Short labels shown on the model cards, e.g. "opset 18".</summary>
    IReadOnlyList<string> Badges { get; }

    /// <summary>Key facts shown in the inspector status bar and structure summary.</summary>
    IReadOnlyList<KeyValuePair<string, string>> Facts { get; }

    ComputationGraph Graph { get; }
    IReadOnlyList<TensorSchema> Inputs { get; }
    IReadOnlyList<TensorSchema> Outputs { get; }

    /// <summary>What <see cref="Initializers"/> are called for this technology ("Initializers", "Learned attributes"...).</summary>
    string WeightsLabel { get; }

    /// <summary>Weights / learned parameters (name, shape, element count).</summary>
    IReadOnlyList<InitializerInfo> Initializers { get; }

    /// <summary>Roots of the structure tree (operators, estimators...). Children are created on demand.</summary>
    IReadOnlyList<StructureNode> Structure { get; }
}