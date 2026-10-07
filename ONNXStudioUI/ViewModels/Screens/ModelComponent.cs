using ONNXStudio.Core.Models;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// View model of a <see cref="StructureNode"/>: the structure tree is built by the model itself
/// (ONNX operators, scikit-learn estimators...), this wrapper only formats entries for display.
/// Children and parameter previews are created on demand.
/// </summary>
public sealed class ModelComponent
{
    private readonly StructureNode _node;
    private readonly Lazy<IReadOnlyList<ModelComponent>> _children;
    private readonly Lazy<IReadOnlyList<AttributeEntry>> _parameters;
    private readonly Lazy<IReadOnlyList<AttributeEntry>> _fitted;

    private ModelComponent(StructureNode node)
    {
        _node = node;
        _children = new(() => node.Children.Select(child => new ModelComponent(child)).ToArray());
        _parameters = new(() => node.Parameters.Select(entry => new AttributeEntry(entry.Key, entry.Value)).ToArray());
        _fitted = new(() => node.Fitted.Select(entry => new AttributeEntry(entry.Key, entry.Value)).ToArray());
    }

    public string Header => _node.Header;
    public string Kind => _node.Kind;
    public string Description => _node.Description;
    public string? NodeId => _node.NodeId;
    public IReadOnlyList<ModelComponent> Children => _children.Value;
    public IReadOnlyList<AttributeEntry> Parameters => _parameters.Value;

    /// <summary>Attributes learned during training (scikit-learn models only).</summary>
    public IReadOnlyList<AttributeEntry> Fitted => _fitted.Value;
    public bool HasFitted => Fitted.Count > 0;

    public static IReadOnlyList<ModelComponent> Build(IModel model) =>
        model.Structure.Select(node => new ModelComponent(node)).ToArray();
}