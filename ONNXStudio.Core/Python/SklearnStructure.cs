using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Python;

/// <summary>Structure tree of a scikit-learn model: estimators, hyper-parameters and learned attributes.</summary>
internal static class SklearnStructure
{
    /// <summary>Root node of the estimator tree. <paramref name="nodeIds"/> links estimators to the graph nodes drawn for them.</summary>
    public static StructureNode Component(PythonModelInfo info, IReadOnlyDictionary<PythonComponent, string>? nodeIds)
    {
        IReadOnlyList<StructureEntry> Overview() => OverviewOf(info);
        if (info.Components is null)
            return new(info.Summary, "scikit-learn", $"{info.Module}.{info.ClassName}",
                () => [.. Overview(), .. info.Parameters.Select(p => new StructureEntry(p.Key, p.Value))]);
        return Node(info.Components, info.Summary, Overview, nodeIds);
    }

    public static IReadOnlyList<StructureEntry> Parameters(PythonComponent component)
    {
        var entries = new List<StructureEntry> { new("Class", Qualified(component)) };
        if (component.Description.Length > 0) entries.Add(new("Details", component.Description));
        entries.AddRange(component.Parameters.Select(p => new StructureEntry(p.Key, p.Value)));
        return entries;
    }

    public static IReadOnlyList<StructureEntry> Fitted(PythonComponent component)
        => component.Fitted.Select(Entry).ToArray();

    /// <summary>A learned attribute as an entry: a paged preview of its values, or its summary when it has none.</summary>
    public static StructureEntry Entry(PythonFittedAttribute attribute)
    {
        if (!attribute.HasValues) return new(attribute.Name, attribute.Summary);
        var shape = attribute.Shape.Count > 0 ? $" · {attribute.DType} ({string.Join(", ", attribute.Shape)})" : string.Empty;
        var preview = attribute.Count > attribute.Values.Count ? $" · first {attribute.Values.Count:N0} of {attribute.Count:N0}" : string.Empty;
        return new(attribute.Name + shape + preview, attribute.Values.ToArray());
    }

    public static string Qualified(PythonComponent component)
        => string.IsNullOrEmpty(component.Module) ? component.ClassName : $"{component.Module}.{component.ClassName}";

    private static StructureNode Node(PythonComponent component, string? header,
        Func<IReadOnlyList<StructureEntry>>? overview, IReadOnlyDictionary<PythonComponent, string>? nodeIds)
    {
        var title = header ?? (component.Name == component.ClassName ? component.ClassName : $"{component.Name} : {component.ClassName}");
        return new(title, component.Kind,
            string.IsNullOrEmpty(component.Description) ? Qualified(component) : $"{Qualified(component)} · {component.Description}",
            () => [.. overview?.Invoke() ?? [], .. Parameters(component)],
            () => component.Children.Select(child => Node(child, null, null, nodeIds)).ToArray(),
            nodeId: nodeIds != null && nodeIds.TryGetValue(component, out var id) ? id : null,
            fitted: () => Fitted(component));
    }

    private static IReadOnlyList<StructureEntry> OverviewOf(PythonModelInfo info)
    {
        var entries = new List<StructureEntry>();
        if (info.FeatureNames is { Count: > 0 }) entries.Add(new("Input features", info.FeatureNames.ToArray()));
        else if (info.FeatureCount is { } count) entries.Add(new("Input features", count));
        if (info.Classes is { Count: > 0 }) entries.Add(new("Classes", info.Classes.ToArray()));
        if (info.Methods.Count > 0) entries.Add(new("Methods", string.Join(", ", info.Methods)));
        if (info.Warnings.Count > 0) entries.Add(new("Warnings", info.Warnings.ToArray()));
        if (info.TrainedWithSklearn != null) entries.Add(new("Trained with scikit-learn", info.TrainedWithSklearn));
        if (info.RuntimeSklearn != null) entries.Add(new("Inspected with scikit-learn", info.RuntimeSklearn));
        if (info.Packages.Count > 0)
            entries.Add(new("Packages", info.Packages.Select(p => $"{p.Key} {p.Value}").ToArray()));
        return entries;
    }
}
