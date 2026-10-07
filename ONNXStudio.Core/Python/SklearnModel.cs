using System.Globalization;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Core.Python;

/// <summary>How the inputs of a scikit-learn model are laid out in its schema.</summary>
public enum SklearnInputLayout
{
    /// <summary>One input "X" [rows, features].</summary>
    Matrix,

    /// <summary>One input [rows, 1] per named column (models with a ColumnTransformer).</summary>
    Columns,

    /// <summary>One text input [rows, 1].</summary>
    Text
}

/// <summary>
/// A scikit-learn model loaded through the Python worker, seen like any other model by the studio:
/// the estimator tree is drawn as a graph, inputs/outputs get a tensor schema, and inference goes
/// through <see cref="SklearnInferenceBackend"/>.
/// </summary>
public sealed class SklearnModel : IModel
{
    private static readonly HashSet<string> CategoricalEncoders = new(StringComparer.Ordinal)
    {
        "OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "LabelEncoder", "CountVectorizer", "TfidfVectorizer", "HashingVectorizer", "FeatureHasher"
    };

    private readonly Lazy<IReadOnlyList<StructureNode>> _structure;
    private readonly SklearnGraph _graph;

    private SklearnModel(PythonModel file, PythonModelInfo info)
    {
        File = file;
        Info = info;
        (InputLayout, Inputs) = BuildInputs(info);
        Outputs = BuildOutputs(info);
        _graph = SklearnGraph.Build(info, Inputs);
        Graph = _graph.Graph;
        Initializers = BuildWeights(info.Components);
        _structure = new(() => BuildStructure());
    }

    public static SklearnModel Create(PythonModel file, PythonModelInfo info) => new(file, info);

    public PythonModel File { get; }
    public PythonModelInfo Info { get; }
    public SklearnInputLayout InputLayout { get; }

    public string Id { get; } = Guid.NewGuid().ToString("N");
    public string Name => Path.GetFileNameWithoutExtension(File.Name);
    public string FilePath => File.FilePath;
    public long FileSize => File.FileSize;
    public string FileSizeDisplay => File.FileSizeDisplay;
    public DateTime LoadedAt { get; } = DateTime.UtcNow;

    public string Format => "scikit-learn";
    public string Producer => "scikit-learn " + (Info.TrainedWithSklearn ?? Info.RuntimeSklearn);
    public string Description => Info.Summary;

    public IReadOnlyList<string> Badges => ["scikit-learn", Info.Components?.Kind.ToLowerInvariant() ?? "estimator"];

    public IReadOnlyList<KeyValuePair<string, string>> Facts
    {
        get
        {
            var facts = new List<KeyValuePair<string, string>> { new("Producer", Producer), new("Estimator", Info.Summary) };
            if (Info.RuntimeSklearn != null && Info.RuntimeSklearn != Info.TrainedWithSklearn && Info.TrainedWithSklearn != null)
                facts.Add(new("Loaded with", "scikit-learn " + Info.RuntimeSklearn));
            return facts;
        }
    }

    public ComputationGraph Graph { get; }
    public IReadOnlyList<TensorSchema> Inputs { get; }
    public IReadOnlyList<TensorSchema> Outputs { get; }
    public string WeightsLabel => "Learned attributes";
    public IReadOnlyList<InitializerInfo> Initializers { get; }
    public IReadOnlyList<StructureNode> Structure => _structure.Value;

    // ----- schema -----

    private static (SklearnInputLayout, IReadOnlyList<TensorSchema>) BuildInputs(PythonModelInfo info)
    {
        if (info.IsTextModel)
            return (SklearnInputLayout.Text, [new TensorSchema("text", DataType.String, [null, 1]) { Description = "One text per row" }]);

        if (info.FeatureNames is { Count: > 0 } names && Descendants(info.Components).Any(c => c.Kind == "ColumnTransformer"))
        {
            var columns = names.Select(name => new TensorSchema(name, IsCategorical(info.Components, name) ? DataType.String : DataType.Float64, [null, 1])).ToArray();
            return (SklearnInputLayout.Columns, columns);
        }

        long? features = info.FeatureCount ?? info.FeatureNames?.Count;
        return (SklearnInputLayout.Matrix, [new TensorSchema("X", DataType.Float64, [null, features])
        {
            Description = info.FeatureNames is { Count: > 0 } ? string.Join(", ", info.FeatureNames) : "Feature values of each row"
        }]);
    }

    private static IReadOnlyList<TensorSchema> BuildOutputs(PythonModelInfo info)
    {
        var classCount = info.Classes?.Count;
        var integerLabels = info.Classes is { Count: > 0 } && info.Classes.All(c => long.TryParse(c, NumberStyles.Integer, CultureInfo.InvariantCulture, out _));
        var outputs = new List<TensorSchema>();
        foreach (var method in info.Methods.Count > 0 ? info.Methods : ["predict"])
        {
            outputs.Add(method switch
            {
                "predict" => new TensorSchema(method, info.IsClassifier ? (integerLabels ? DataType.Int64 : DataType.String) : DataType.Float64, [null]),
                "predict_proba" or "predict_log_proba" => new TensorSchema(method, DataType.Float64, [null, classCount]),
                "decision_function" => new TensorSchema(method, DataType.Float64, classCount > 2 ? [null, classCount] : [null]),
                _ => new TensorSchema(method, DataType.Float64, [null, null])
            });
        }
        return outputs;
    }

    private static IEnumerable<PythonComponent> Descendants(PythonComponent? root)
    {
        if (root is null) yield break;
        yield return root;
        foreach (var child in root.Children)
            foreach (var descendant in Descendants(child)) yield return descendant;
    }

    // A column is categorical when a transformer that encodes categories is applied to it.
    private static bool IsCategorical(PythonComponent? root, string column)
        => Descendants(root).Where(c => c.Kind == "ColumnTransformer")
            .SelectMany(c => c.Children)
            .Any(branch => branch.Description.Contains($"'{column}'", StringComparison.Ordinal)
                           && Descendants(branch).Any(d => CategoricalEncoders.Contains(d.ClassName)));

    private static IReadOnlyList<InitializerInfo> BuildWeights(PythonComponent? root)
    {
        var weights = new List<InitializerInfo>();
        foreach (var component in Descendants(root))
        {
            foreach (var attribute in component.Fitted.Where(a => a.Kind is "array" or "sparse" && a.Shape.Count > 0))
                weights.Add(new InitializerInfo($"{component.Name}.{attribute.Name}", attribute.Shape.Select(d => (long)d).ToArray(),
                    attribute.Shape.Aggregate(1L, (a, b) => a * b)));
        }
        return weights;
    }

    // ----- structure -----

    private IReadOnlyList<StructureNode> BuildStructure()
    {
        var tree = SklearnStructure.Component(Info, _graph.NodeIds);
        return
        [
            new(Name, "Model", $"scikit-learn model loaded from {File.Name}.",
                () => [new("Format", Format), new("Producer", Producer), new("Estimator", Info.Summary), new("File", FilePath)],
                () =>
                [
                    StructureBuilder.Section("Inputs", Inputs, StructureBuilder.Tensor),
                    tree,
                    StructureBuilder.Section("Outputs", Outputs, StructureBuilder.Tensor)
                ])
        ];
    }
}