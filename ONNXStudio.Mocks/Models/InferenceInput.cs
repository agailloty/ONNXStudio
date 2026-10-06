using System;
using System.Collections.Generic;

namespace ONNXStudio.Mocks.Models;

/// <summary>
/// Represents an input for model inference
/// </summary>
public class InferenceInput
{
    public string Name { get; set; } = string.Empty;
    public string DisplayName { get; set; } = string.Empty;
    public string Type { get; set; } = "float32";
    public long[] Shape { get; set; } = Array.Empty<long>();
    public object? Value { get; set; }
    public object? DefaultValue { get; set; }
    
    // UI properties
    public bool IsRequired { get; set; } = true;
    public bool IsValid { get; set; } = true;
    public string? ErrorMessage { get; set; }
    
    // Type-specific properties
    public string InputType { get; set; } = "Number"; // Number, Image, Text, Slider, Vector, Dropdown
    
    public string ShapeDisplay => string.Join("x", Shape);
    
    // For image inputs
    public string? ImagePath { get; set; }
    public int ExpectedWidth => Shape.Length > 3 ? (int)Shape[^2] : 224;
    public int ExpectedHeight => Shape.Length > 3 ? (int)Shape[^3] : 224;
    public int ExpectedChannels => Shape.Length > 3 ? (int)Shape[^4] : 3;
    
    // For number inputs
    public double Min { get; set; } = double.MinValue;
    public double Max { get; set; } = double.MaxValue;
    public double Step { get; set; } = 1.0;
    
    // For dropdown inputs
    public List<string> Options { get; set; } = new();
    public int SelectedIndex { get; set; }
    
    // For vector inputs
    public string VectorValues { get; set; } = string.Empty;
}

/// <summary>
/// Represents inference output
/// </summary>
public class InferenceOutput
{
    public string Name { get; set; } = string.Empty;
    public string DisplayName { get; set; } = string.Empty;
    public string Type { get; set; } = "float32";
    public long[] Shape { get; set; } = Array.Empty<long>();
    public object? Value { get; set; }

    public string ShapeDisplay => string.Join("x", Shape);

    // Human-friendly value for regression outputs (e.g. "$205,300")
    public string DisplayValue { get; set; } = string.Empty;

    // For classification outputs
    public List<ClassificationResult> ClassificationResults { get; set; } = new();
}

/// <summary>
/// Represents a classification result
/// </summary>
public class ClassificationResult
{
    public string Label { get; set; } = string.Empty;
    public double Confidence { get; set; }
    public int Index { get; set; }
}

/// <summary>
/// Represents inference result
/// </summary>
public class InferenceResult
{
    public string ModelId { get; set; } = string.Empty;
    public string ModelName { get; set; } = string.Empty;
    public DateTime ExecutionTime { get; set; } = DateTime.Now;
    public long ExecutionTimeMs => (long)(DateTime.Now - ExecutionTime).TotalMilliseconds;
    public List<InferenceOutput> Outputs { get; set; } = new();
    public bool IsSuccess { get; set; } = true;
    public string? ErrorMessage { get; set; }
}
