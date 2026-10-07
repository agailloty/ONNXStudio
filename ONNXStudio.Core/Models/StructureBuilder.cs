namespace ONNXStudio.Core.Models;

/// <summary>Helpers shared by the structure trees of every model type.</summary>
internal static class StructureBuilder
{
    public const int BranchSize = 100;

    public static StructureNode Tensor(TensorSchema tensor) => new(tensor.Name, "Tensor", tensor.Display,
        () => [new("Type", tensor.Type.ToDisplayName()), new("Shape", tensor.Shape.ToArray()), new("Description", tensor.Description)]);

    /// <summary>A group node listing <paramref name="items"/>, split in bounded branches when it is large.</summary>
    public static StructureNode Section<T>(string name, IReadOnlyList<T> items, Func<T, StructureNode> create) =>
        new($"{name} ({items.Count:N0})", "Group", $"{items.Count:N0} {name.ToLowerInvariant()}",
            children: () => Branch(items, create, 0, items.Count));

    private static IReadOnlyList<StructureNode> Branch<T>(IReadOnlyList<T> items,
        Func<T, StructureNode> create, int start, int count)
    {
        var children = new List<StructureNode>();
        if (count <= BranchSize)
        {
            for (int i = start; i < start + count; i++) children.Add(create(items[i]));
        }
        else
        {
            int size = BranchSize;
            while ((count - 1) / size >= BranchSize) size *= BranchSize;
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