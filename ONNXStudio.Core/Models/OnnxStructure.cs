using ONNXStudio.Core.Python;

namespace ONNXStudio.Core.Models;

/// <summary>Structure tree of an ONNX model: tensors, operators, initializers and, for converted models, the scikit-learn origin.</summary>
internal static class OnnxStructure
{
    public static IReadOnlyList<StructureNode> Build(OnnxModel model)
    {
        var roots = new List<StructureNode> { Root(model) };
        if (model.Metadata.TryGetValue(PythonModelInfo.OnnxMetadataKey, out var json) && PythonModelInfo.TryParse(json) is { } source)
            roots.Add(SklearnStructure.Component(source, null));
        return roots;
    }

    private static StructureNode Root(OnnxModel model) =>
        new(model.Name, "Model", "Explore the model's ONNX operators, tensors and attributes.",
            () =>
            {
                var parameters = new List<StructureEntry>
                {
                    new("Producer", model.ProducerName), new("Domain", model.Domain),
                    new("Opset", model.OpsetVersion), new("Version", model.ModelVersion),
                    new("IR version", model.IrVersion), new("Description", model.DocString)
                };
                parameters.AddRange(model.Metadata
                    .Where(entry => entry.Key != PythonModelInfo.OnnxMetadataKey)
                    .Select(entry => new StructureEntry(entry.Key, entry.Value)));
                return parameters;
            },
            () =>
            [
                StructureBuilder.Section("Inputs", model.Inputs, StructureBuilder.Tensor),
                StructureBuilder.Section("Operators", model.Graph.Nodes, Operator),
                StructureBuilder.Section("Outputs", model.Outputs, StructureBuilder.Tensor),
                StructureBuilder.Section("Initializers", model.Initializers, weight => new(weight.Name, "Initializer",
                    "Weight tensor metadata", () => [new("Shape", weight.Shape.ToArray()), new("Elements", weight.ElementCount)]))
            ]);

    private static StructureNode Operator(GraphNode node) => new(
        string.IsNullOrEmpty(node.Name) ? node.OpType : $"{node.Name} : {node.OpType}",
        "Operator", $"{node.OpType} · {node.Attributes.Count} attributes", () =>
        {
            var parameters = new List<StructureEntry>
            {
                new("Operator", node.OpType), new("Domain", node.Domain),
                new("Inputs", node.InputIds), new("Outputs", node.OutputIds)
            };
            parameters.AddRange(node.Attributes.Select(a => new StructureEntry(a.Key, a.Value)));
            return parameters;
        }, nodeId: node.Id);
}