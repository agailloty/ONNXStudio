using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Services;

/// <summary>
/// Aggregated graph statistics for the inspector header (US-002).
/// </summary>
public sealed class GraphStatistics
{
    public int NodeCount { get; init; }
    public int EdgeCount { get; init; }
    public int InitializerCount { get; init; }
    public long TotalParameters { get; init; }
    public int InputCount { get; init; }
    public int OutputCount { get; init; }
    public int DistinctOpTypes { get; init; }
}

/// <summary>
/// Producers and consumers of a node (US-002 dependency view).
/// </summary>
public sealed class NodeDependencies
{
    public string NodeId { get; }
    public IReadOnlyList<GraphNode> DependsOn { get; }
    public IReadOnlyList<GraphNode> DependedBy { get; }

    public NodeDependencies(string nodeId, IReadOnlyList<GraphNode> dependsOn, IReadOnlyList<GraphNode> dependedBy)
    {
        NodeId = nodeId;
        DependsOn = dependsOn;
        DependedBy = dependedBy;
    }
}

/// <summary>
/// Graph exploration queries (US-002): statistics, search, filter by
/// operator category and dependency traversal.
/// </summary>
public interface IGraphAnalysisService
{
    GraphStatistics GetStatistics(OnnxModel model);
    IReadOnlyList<GraphNode> Search(OnnxModel model, string? query, string? category);
    IReadOnlyList<string> GetCategories(OnnxModel model);
    string GetCategory(string opType);
    NodeDependencies GetDependencies(OnnxModel model, string nodeId);
}

public sealed class GraphAnalysisService : IGraphAnalysisService
{
    public GraphStatistics GetStatistics(OnnxModel model) => new()
    {
        NodeCount = model.Graph.Nodes.Count,
        EdgeCount = model.Graph.Edges.Count,
        InitializerCount = model.Initializers.Count,
        TotalParameters = model.Initializers.Sum(i => i.ElementCount),
        InputCount = model.Inputs.Count,
        OutputCount = model.Outputs.Count,
        DistinctOpTypes = model.Graph.Nodes.Select(n => n.OpType).Distinct().Count()
    };

    public IReadOnlyList<GraphNode> Search(OnnxModel model, string? query, string? category)
    {
        IEnumerable<GraphNode> nodes = model.Graph.Nodes;

        if (!string.IsNullOrWhiteSpace(query))
        {
            nodes = nodes.Where(n =>
                n.Name.Contains(query, StringComparison.OrdinalIgnoreCase) ||
                n.OpType.Contains(query, StringComparison.OrdinalIgnoreCase));
        }

        if (!string.IsNullOrEmpty(category) && category != "All")
        {
            nodes = nodes.Where(n => string.Equals(GetCategory(n.OpType), category, StringComparison.OrdinalIgnoreCase));
        }

        return nodes.ToList();
    }

    public IReadOnlyList<string> GetCategories(OnnxModel model)
    {
        var categories = new List<string> { "All" };
        categories.AddRange(model.Graph.Nodes
            .Select(n => GetCategory(n.OpType))
            .Distinct()
            .OrderBy(c => c));
        return categories;
    }

    /// <summary>
    /// Groups operator types into UI categories (color-coded by the viewer).
    /// </summary>
    public string GetCategory(string opType) => opType switch
    {
        "Conv" or "ConvTranspose" or "QLinearConv" => "Conv",
        "MaxPool" or "AveragePool" or "GlobalAveragePool" or "GlobalMaxPool" => "Pool",
        "Relu" or "LeakyRelu" or "Sigmoid" or "Tanh" or "Softmax" or "Elu" or "Gelu" => "Activation",
        "Gemm" or "MatMul" or "Add" or "Sub" or "Mul" or "Div" => "Linear",
        "BatchNormalization" or "LayerNormalization" or "InstanceNormalization" or "GroupNormalization" => "Normalization",
        "Scaler" or "TreeEnsembleClassifier" or "TreeEnsembleRegressor" or "LinearClassifier" or "LinearRegressor" => "Pipeline",
        _ => "Other"
    };

    public NodeDependencies GetDependencies(OnnxModel model, string nodeId)
    {
        var nodeById = model.Graph.Nodes.ToDictionary(n => n.Id);
        var dependsOn = model.Graph.Edges
            .Where(e => e.ToNodeId == nodeId && nodeById.ContainsKey(e.FromNodeId))
            .Select(e => nodeById[e.FromNodeId])
            .Distinct()
            .ToList();

        var dependedBy = model.Graph.Edges
            .Where(e => e.FromNodeId == nodeId && nodeById.ContainsKey(e.ToNodeId))
            .Select(e => nodeById[e.ToNodeId])
            .Distinct()
            .ToList();

        return new NodeDependencies(nodeId, dependsOn, dependedBy);
    }
}
