using System;
using System.Collections.Generic;
using System.Linq;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using ONNXStudioUI.ViewModels.Screens;

namespace ONNXStudioUI.Views.Screens;

public partial class WelcomeView : UserControl
{
    public WelcomeView()
    {
        InitializeComponent();

        AddHandler(DragDrop.DragOverEvent, OnDragOver);
        AddHandler(DragDrop.DropEvent, OnDrop);
    }

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        e.DragEffects = GetOnnxFiles(e).Count > 0 ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
    }

    private void OnDrop(object? sender, DragEventArgs e)
    {
        e.Handled = true;
        if (DataContext is not WelcomeViewModel vm)
        {
            return;
        }

        var path = GetOnnxFiles(e).FirstOrDefault();
        if (path != null)
        {
            vm.DropModelCommand.Execute(path);
        }
    }

    private static IReadOnlyList<string> GetOnnxFiles(DragEventArgs e)
    {
        var paths = new List<string>();
        foreach (var item in e.DataTransfer.Items)
        {
            if (!item.Formats.Contains(DataFormat.File))
            {
                continue;
            }

            if (item.TryGetRaw(DataFormat.File) is IStorageItem storage)
            {
                var path = storage.TryGetLocalPath() ?? storage.Path.AbsolutePath;
                if (path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
                {
                    paths.Add(path);
                }
            }
        }
        return paths;
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
