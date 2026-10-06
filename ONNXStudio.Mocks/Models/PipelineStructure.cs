using System.Collections.Generic;
using System.Linq;

namespace ONNXStudio.Mocks.Models;

/// <summary>
/// A step of a (scikit-learn style) pipeline, with its fitted/constructor parameters.
/// Used by the structure explorer (graph-free model exploration).
/// </summary>
public class PipelineComponent
{
    public string StepName { get; set; } = string.Empty;   // e.g. "scaler"
    public string Name { get; set; } = string.Empty;       // e.g. "StandardScaler"
    public string Kind { get; set; } = string.Empty;       // pipeline / preprocessing / regressor / layer-group
    public string Icon { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public List<ParameterEntry> Parameters { get; set; } = new();
    public List<PipelineComponent> Children { get; set; } = new();

    public string Header => string.IsNullOrEmpty(StepName) ? Name : StepName + " : " + Name;
    public int ParameterCount => Parameters.Count + Children.Sum(c => c.ParameterCount);
}

/// <summary>
/// A single model parameter (key/value), sklearn get_params() style.
/// </summary>
public class ParameterEntry
{
    public string Key { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
    public string ValueType { get; set; } = "str"; // str / bool / number / array

    public bool IsNumber => ValueType == "number";
}

/// <summary>
/// Feature importance (for models that expose them, e.g. tree-based regressors).
/// </summary>
public class FeatureImportance
{
    public string Feature { get; set; } = string.Empty;
    public double Importance { get; set; }

    public string PercentDisplay => (Importance * 100).ToString("0.0") + "%";
}
