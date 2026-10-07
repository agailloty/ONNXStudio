using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Python;

/// <summary>
/// Draws a scikit-learn estimator tree as a computation graph: pipeline steps are chained,
/// column transformers / unions / voting ensembles fan out and merge again.
/// </summary>
internal sealed class SklearnGraph
{
    private readonly List<GraphNode> _nodes = new();
    private readonly List<GraphEdge> _edges = new();
    private readonly Dictionary<string, string> _producers = new();
    private readonly Dictionary<PythonComponent, string> _ids = new(ReferenceEqualityComparer.Instance);

    public ComputationGraph Graph => new(_nodes, _edges);

    /// <summary>Graph node drawn for each estimator (for linking the structure tree to the graph).</summary>
    public IReadOnlyDictionary<PythonComponent, string> NodeIds => _ids;

    public static SklearnGraph Build(PythonModelInfo info, IReadOnlyList<TensorSchema> inputs)
    {
        var graph = new SklearnGraph();
        var inputNames = inputs.Select(i => i.Name).ToArray();
        if (info.Components is { } root) graph.Emit(root, inputNames);
        else graph.AddNode(info.ClassName, info.ClassName, info.Module, inputNames, info.IsClassifier ? "Classifier" : "Estimator", new Dictionary<string, object>());
        return graph;
    }

    private enum Role { Step, Branch, Inner }

    // Attribute-backed children (estimator_, best_estimator_...) are details of their parent, not stages of the data flow.
    private static Role RoleOf(PythonComponent parent, PythonComponent child)
        => parent.Kind == "Pipeline" ? Role.Step
            : parent.Kind is "ColumnTransformer" or "FeatureUnion" ? Role.Branch
            : child.Name == "best_estimator_" ? Role.Step
            : child.Name.EndsWith('_') ? Role.Inner
            : Role.Branch;

    private string[] Emit(PythonComponent component, string[] inputs)
    {
        var steps = component.Children.Where(c => RoleOf(component, c) == Role.Step).ToList();
        if (steps.Count > 0)
        {
            var current = inputs;
            foreach (var step in steps) current = Emit(step, current);
            return current;
        }

        var branches = component.Children.Where(c => RoleOf(component, c) == Role.Branch).ToList();
        var merged = branches.Count > 0 ? branches.SelectMany(b => Emit(b, inputs)).ToArray() : inputs;
        var attributes = new Dictionary<string, object>();
        foreach (var entry in SklearnStructure.Parameters(component).Concat(SklearnStructure.Fitted(component)))
            attributes[entry.Key] = entry.Value;
        var kind = component.Kind;
        var category = branches.Count > 0 || kind is "ColumnTransformer" or "FeatureUnion" ? "Pipeline"
            : kind is "Transformer" or "Classifier" or "Regressor" ? kind
            : kind == "Passthrough" ? "Other" : "Estimator";
        var id = AddNode(component.Name, component.ClassName, component.Module, merged, category, attributes);
        _ids[component] = id;
        return _nodes[^1].OutputIds.ToArray();
    }

    private string AddNode(string name, string opType, string module, string[] inputs, string category, Dictionary<string, object> attributes)
    {
        var id = $"node_{_nodes.Count}";
        var output = $"{name}:{id}";
        _nodes.Add(new GraphNode(id, name, opType, module.Split('.')[0], attributes, inputs, [output], category));
        foreach (var input in inputs.Distinct())
        {
            if (_producers.TryGetValue(input, out var from)) _edges.Add(new GraphEdge(input, from, id));
        }
        _producers[output] = id;
        return id;
    }
}
