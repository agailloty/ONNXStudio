using ONNXStudio.Core.Models;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// Structure explorer inspired by the mock's PipelineComponent. Only actual
/// ONNX metadata is shown: original sklearn classes/parameters may not survive export.
/// Children and parameter previews are created on demand, with bounded branch sizes.
/// </summary>
public sealed class ModelComponent
{
    private readonly Lazy<IReadOnlyList<ModelComponent>> _children;
    private readonly Lazy<IReadOnlyList<AttributeEntry>> _parameters;
    public string Header { get; }
    public string Kind { get; }
    public string Description { get; }
    public string? NodeId { get; }
    public IReadOnlyList<ModelComponent> Children => _children.Value;
    public IReadOnlyList<AttributeEntry> Parameters => _parameters.Value;

    private ModelComponent(string header, string kind, string description,
        Func<IReadOnlyList<AttributeEntry>>? parameters = null,
        Func<IReadOnlyList<ModelComponent>>? children = null, string? nodeId = null)
    {
        Header = header;
        Kind = kind;
        Description = description;
        NodeId = nodeId;
        _parameters = new(parameters ?? (() => []));
        _children = new(children ?? (() => []));
    }

    public static IReadOnlyList<ModelComponent> Build(OnnxModel model) =>
    [
        new(model.Name, "Model", "Explore the model's ONNX operators, tensors and attributes.",
            () => [new("Producer", model.ProducerName), new("Domain", model.Domain),
                new("Opset", model.OpsetVersion), new("Version", model.ModelVersion),
                new("IR version", model.IrVersion), new("Description", model.DocString)],
            () =>
            [
                Section("Inputs", model.Inputs, Tensor),
                Section("Operators", model.Graph.Nodes, Node),
                Section("Outputs", model.Outputs, Tensor),
                Section("Initializers", model.Initializers, weight => new(weight.Name, "Initializer",
                    "Weight tensor metadata", () => [new("Shape", weight.Shape.ToArray()), new("Elements", weight.ElementCount)]))
            ])
    ];

    private static ModelComponent Tensor(TensorSchema tensor) => new(tensor.Name, "Tensor", tensor.Display,
        () => [new("Type", tensor.Type.ToDisplayName()), new("Shape", tensor.Shape.ToArray()), new("Description", tensor.Description)]);

    private static ModelComponent Node(GraphNode node) => new(
        string.IsNullOrEmpty(node.Name) ? node.OpType : $"{node.Name} : {node.OpType}",
        "Operator", $"{node.OpType} · {node.Attributes.Count} attributes", () =>
        {
            var parameters = new List<AttributeEntry>
            {
                new("Operator", node.OpType), new("Domain", node.Domain),
                new("Inputs", node.InputIds), new("Outputs", node.OutputIds)
            };
            parameters.AddRange(node.Attributes.Select(a => new AttributeEntry(a.Key, a.Value)));
            return parameters;
        }, nodeId: node.Id);

    private static ModelComponent Section<T>(string name, IReadOnlyList<T> items, Func<T, ModelComponent> create) =>
        new($"{name} ({items.Count:N0})", "Group", $"{items.Count:N0} {name.ToLowerInvariant()}",
            children: () => Branch(items, create, 0, items.Count));

    private static IReadOnlyList<ModelComponent> Branch<T>(IReadOnlyList<T> items,
        Func<T, ModelComponent> create, int start, int count)
    {
        const int branchSize = 100;
        var children = new List<ModelComponent>();
        if (count <= branchSize)
        {
            for (int i = start; i < start + count; i++) children.Add(create(items[i]));
        }
        else
        {
            int size = branchSize;
            while ((count - 1) / size >= branchSize) size *= branchSize;
            for (int i = start; i < start + count; i += size)
            {
                int offset = i, length = Math.Min(size, start + count - i);
                children.Add(new($"{offset + 1:N0}–{offset + length:N0}", "Group", $"{length:N0} items",
                    children: () => Branch(items, create, offset, length)));
            }
        }
        return children;
    }
}
