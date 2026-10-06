using System;
using System.Collections.Generic;

namespace ONNXStudio.Mocks.Models;

/// <summary>
/// Represents a loaded ONNX model
/// </summary>
public class OnnxModel
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = string.Empty;
    public string FilePath { get; set; } = string.Empty;
    public string FileName => System.IO.Path.GetFileName(FilePath);
    public long FileSize { get; set; }
    public DateTime LoadedAt { get; set; } = DateTime.Now;
    
    // Model metadata
    public string Producer { get; set; } = "Unknown";
    public int OpsetVersion { get; set; }
    public string IrVersion { get; set; } = "8";
    public string ModelVersion { get; set; } = "1";
    public string DocString { get; set; } = string.Empty;
    
    // Graph data
    public ComputationGraph Graph { get; set; } = new();
    
    // Size info
    public string SizeDisplay => FormatFileSize(FileSize);
    public string LoadedTimeDisplay => (DateTime.Now - LoadedAt).TotalMinutes < 1 
        ? "Just now"
        : $"{(int)(DateTime.Now - LoadedAt).TotalMinutes}m ago";
    
    // Preview
    public string PreviewIcon => GetPreviewIcon();
    
    // Model type classification
    public ModelType Type => ClassifyModel();
    
    // API endpoint
    public string ApiEndpoint => $"/models/{Name}/predict";
    
    private static string FormatFileSize(long bytes)
    {
        string[] sizes = { "B", "KB", "MB", "GB" };
        int order = 0;
        double len = bytes;
        while (len >= 1024 && order < sizes.Length - 1)
        {
            order++;
            len /= 1024;
        }
        return $"{len:0.#} {sizes[order]}";
    }
    
    private string GetPreviewIcon()
    {
        return Type switch
        {
            ModelType.CNN => "🖼️",
            ModelType.RNN or ModelType.LSTM => "📜",
            ModelType.Transformer => "🤖",
            ModelType.Classifier => "🏷️",
            ModelType.Detector => "🔍",
            _ => "📦"
        };
    }
    
    private ModelType ClassifyModel()
    {
        // Simple classification based on graph structure
        if (Graph.Outputs.Count > 0)
        {
            var outputShape = Graph.Outputs[0].Shape;
            if (outputShape.Length > 1 && outputShape[^1] > 10)
                return ModelType.Classifier;
        }
        
        bool hasConv = Graph.Nodes.Exists(n => n.OpType.Contains("Conv"));
        bool hasPool = Graph.Nodes.Exists(n => n.OpType.Contains("Pool"));
        bool hasRNN = Graph.Nodes.Exists(n => n.OpType.Contains("LSTM") || n.OpType.Contains("GRU") || n.OpType.Contains("RNN"));
        bool hasAttention = Graph.Nodes.Exists(n => n.OpType.Contains("Attention") || n.OpType.Contains("MatMul"));
        
        if (hasAttention && hasRNN)
            return ModelType.Transformer;
        if (hasRNN)
            return ModelType.RNN;
        if (hasConv && hasPool)
            return ModelType.CNN;
        
        return ModelType.Other;
    }
}

public enum ModelType
{
    CNN,
    RNN,
    LSTM,
    Transformer,
    Classifier,
    Detector,
    Segmenter,
    Other
}
