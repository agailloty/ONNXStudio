using System;
using System.Collections.Generic;
using Avalonia;
using Avalonia.Styling;

namespace ONNXStudio.Mocks.Models;

/// <summary>
/// Represents a node in the ONNX computation graph
/// </summary>
public class GraphNode
{
    public string Name { get; set; } = string.Empty;
    public string OpType { get; set; } = string.Empty;
    public string? Domain { get; set; }
    public int[]? InputIndices { get; set; }
    public int[]? OutputIndices { get; set; }
    public Dictionary<string, object> Attributes { get; set; } = new();
    public List<GraphTensor> Inputs { get; set; } = new();
    public List<GraphTensor> Outputs { get; set; } = new();
    
    // UI State
    public bool IsSelected { get; set; }
    public bool IsHovered { get; set; }
    
    // Position for graph visualization
    public double X { get; set; }
    public double Y { get; set; }
    
    // Color based on OpType category
    public string ColorHex => GetNodeColor();

    public string GetNodeColor()
    {
        bool light = Application.Current?.ActualThemeVariant == ThemeVariant.Light;
        return OpType switch
        {
            "Conv" or "ConvTranspose" => light ? "#0969DA" : "#58A6FF", // Blue
            "MaxPool" or "AveragePool" or "GlobalAveragePool" => light ? "#1A7F37" : "#3FB950", // Green
            "Relu" or "Sigmoid" or "Tanh" or "LeakyRelu" or "Softmax" => light ? "#9A6700" : "#F8E454", // Yellow
            "Gemm" or "MatMul" or "Add" or "Sub" or "Mul" or "Div" => light ? "#8250DF" : "#A371F7", // Purple
            "BatchNormalization" or "LayerNormalization" => light ? "#1B7C83" : "#56D3D1", // Teal
            _ => light ? "#57606A" : "#8B949E" // Gray
        };
    }
}

/// <summary>
/// Represents a tensor in the ONNX graph
/// </summary>
public class GraphTensor
{
    public string Name { get; set; } = string.Empty;
    public string DataType { get; set; } = "float32";
    public long[] Shape { get; set; } = Array.Empty<long>();
    public string? ValueInfo { get; set; }

    public string ShapeDisplay => string.Join("x", Shape);
}

/// <summary>
/// Represents the ONNX computation graph
/// </summary>
public class ComputationGraph
{
    public string Name { get; set; } = string.Empty;
    public List<GraphNode> Nodes { get; set; } = new();
    public List<GraphTensor> Inputs { get; set; } = new();
    public List<GraphTensor> Outputs { get; set; } = new();
    public Dictionary<string, GraphTensor> ValueInfo { get; set; } = new();
    
    // Metadata
    public string Producer { get; set; } = "Unknown";
    public int OpsetVersion { get; set; }
    public string IrVersion { get; set; } = "8";
    public long ParameterCount { get; set; }
    public long NodeCount => Nodes.Count;
}
