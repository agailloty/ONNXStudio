namespace ONNXStudio.Core.Models;

/// <summary>
/// A model weight/initializer (name, shape, element count).
/// Raw values are lazy-loaded only if the user opens the weight explorer.
/// </summary>
public sealed class InitializerInfo
{
    public string Name { get; }
    public IReadOnlyList<long> Shape { get; }
    public long ElementCount { get; }

    public InitializerInfo(string name, IReadOnlyList<long> shape, long elementCount)
    {
        Name = name;
        Shape = shape;
        ElementCount = elementCount;
    }

    public string ShapeDisplay => string.Join("x", Shape);
}
