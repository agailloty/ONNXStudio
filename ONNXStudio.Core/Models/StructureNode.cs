namespace ONNXStudio.Core.Models;

/// <summary>One named value of a structure node. A list value is shown page by page.</summary>
public sealed record StructureEntry(string Key, object Value);

/// <summary>
/// UI-agnostic node of the structure tree of a model. Parameters and children are
/// created on demand: a tree ensemble can hold millions of values.
/// </summary>
public sealed class StructureNode
{
    private readonly Lazy<IReadOnlyList<StructureEntry>> _parameters;
    private readonly Lazy<IReadOnlyList<StructureEntry>> _fitted;
    private readonly Lazy<IReadOnlyList<StructureNode>> _children;

    public StructureNode(string header, string kind, string description,
        Func<IReadOnlyList<StructureEntry>>? parameters = null,
        Func<IReadOnlyList<StructureNode>>? children = null,
        string? nodeId = null,
        Func<IReadOnlyList<StructureEntry>>? fitted = null)
    {
        Header = header;
        Kind = kind;
        Description = description;
        NodeId = nodeId;
        _parameters = new(parameters ?? (() => []));
        _fitted = new(fitted ?? (() => []));
        _children = new(children ?? (() => []));
    }

    public string Header { get; }
    public string Kind { get; }
    public string Description { get; }

    /// <summary>Id of the matching <see cref="GraphNode"/>, when the node is drawn in the graph.</summary>
    public string? NodeId { get; }

    public IReadOnlyList<StructureEntry> Parameters => _parameters.Value;

    /// <summary>Attributes learned during training (scikit-learn models).</summary>
    public IReadOnlyList<StructureEntry> Fitted => _fitted.Value;

    public IReadOnlyList<StructureNode> Children => _children.Value;
}