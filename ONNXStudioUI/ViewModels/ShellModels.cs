using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using ONNXStudio.Core.Models;

namespace ONNXStudioUI.ViewModels;

/// <summary>An open screen shown in the editor tab strip of the shell.</summary>
public sealed partial class EditorTab : ObservableObject
{
    public EditorTab(string key, string title, string icon)
    {
        Key = key;
        _title = title;
        Icon = icon;
    }

    /// <summary>"dashboard", "settings" or "&lt;kind&gt;:&lt;modelId&gt;".</summary>
    public string Key { get; }

    /// <summary>Icon resource key (see Resources/Icons.axaml).</summary>
    public string Icon { get; }

    [ObservableProperty]
    private string _title;

    [ObservableProperty]
    private bool _isActive;
}

/// <summary>Node of the side bar model explorer: a loaded model or one of its screens.</summary>
public sealed partial class ExplorerNode : ObservableObject
{
    public ExplorerNode(string key, string header, string icon, OnnxModel model, string? detail = null)
    {
        Key = key;
        Header = header;
        Icon = icon;
        Model = model;
        Detail = detail;
    }

    /// <summary>"&lt;kind&gt;:&lt;modelId&gt;" with kind in model, inspector, playground, apiconfig, sandbox.</summary>
    public string Key { get; }

    public string Header { get; }

    public string Icon { get; }

    public string? Detail { get; }

    public OnnxModel Model { get; }

    public ObservableCollection<ExplorerNode> Children { get; } = new();

    [ObservableProperty]
    private bool _isExpanded = true;
}
