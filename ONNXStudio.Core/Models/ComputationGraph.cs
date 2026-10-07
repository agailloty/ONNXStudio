namespace ONNXStudio.Core.Models;

/// <summary>
/// One operation of the computation graph.
/// </summary>
public sealed class GraphNode
{
    public string Id { get; }
    public string Name { get; }
    public string OpType { get; }
    public string Domain { get; }
    public IReadOnlyDictionary<string, object> Attributes { get; }
    public IReadOnlyList<string> InputIds { get; }
    public IReadOnlyList<string> OutputIds { get; }

    /// <summary>Display category chosen by the model type; when null it is derived from <see cref="OpType"/>.</summary>
    public string? Category { get; }

    public GraphNode(
        string id,
        string name,
        string opType,
        string domain,
        IReadOnlyDictionary<string, object> attributes,
        IReadOnlyList<string> inputIds,
        IReadOnlyList<string> outputIds,
        string? category = null)
    {
        Id = id;
        Name = name;
        OpType = opType;
        Domain = domain;
        Attributes = attributes;
        InputIds = inputIds;
        OutputIds = outputIds;
        Category = category;
    }
}

/// <summary>
/// Tensor flow between two nodes (from one node output to another node input).
/// </summary>
public sealed class GraphEdge
{
    public string TensorName { get; }
    public string FromNodeId { get; }
    public string ToNodeId { get; }

    public GraphEdge(string tensorName, string fromNodeId, string toNodeId)
    {
        TensorName = tensorName;
        FromNodeId = fromNodeId;
        ToNodeId = toNodeId;
    }
}

/// <summary>
/// The full computation graph of a model.
/// </summary>
public sealed class ComputationGraph
{
    public IReadOnlyList<GraphNode> Nodes { get; }
    public IReadOnlyList<GraphEdge> Edges { get; }

    public ComputationGraph(IReadOnlyList<GraphNode> nodes, IReadOnlyList<GraphEdge> edges)
    {
        Nodes = nodes;
        Edges = edges;
    }
}
