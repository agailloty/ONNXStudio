using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Platform.Storage;
using ONNXStudioUI.ViewModels;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DragDrop.SetAllowDrop(this, true);
        AddHandler(DragDrop.DragOverEvent, (_, e) =>
        {
            e.DragEffects = GetFiles(e).Any() ? DragDropEffects.Copy : DragDropEffects.None;
            e.Handled = true;
        });
        AddHandler(DragDrop.DropEvent, async (_, e) =>
        {
            e.Handled = true;
            if (App.Services.GetService(typeof(IModelLoadCoordinator)) is IModelLoadCoordinator loader)
                await loader.LoadManyAsync(GetFiles(e).ToArray());
        });
        KeyDown += async (_, e) =>
        {
            var modifier = OperatingSystem.IsMacOS() ? KeyModifiers.Meta : KeyModifiers.Control;
            if (e.Key != Key.O || !e.KeyModifiers.HasFlag(modifier)) return;
            e.Handled = true;
            if (App.Services.GetService(typeof(IFilePickerService)) is IFilePickerService picker &&
                App.Services.GetService(typeof(IModelLoadCoordinator)) is IModelLoadCoordinator loader)
                await loader.LoadManyAsync(await picker.PickModelFilesAsync());
        };

        DataContextChanged += (_, _) =>
        {
            if (DataContext is MainWindowViewModel shell)
            {
                shell.PropertyChanged += (_, e) =>
                {
                    if (e.PropertyName == nameof(MainWindowViewModel.IsSidebarVisible)) ApplySidebarVisibility(shell.IsSidebarVisible);
                };
                ApplySidebarVisibility(shell.IsSidebarVisible);
            }
        };

        // Wire the storage picker once the window (TopLevel) is created.
        if (App.Services.GetService(typeof(FilePickerService)) is FilePickerService filePicker)
        {
            filePicker.Initialize(() => StorageProvider);
        }
    }

    private double _sidebarWidth = 260;

    // The side bar column is collapsed (not just hidden) so the editor reclaims the space.
    private void ApplySidebarVisibility(bool visible)
    {
        if (this.FindControl<Grid>("Workbench") is not { } workbench) return;
        var column = workbench.ColumnDefinitions[1];
        if (visible)
        {
            column.MinWidth = 180;
            column.MaxWidth = 520;
            column.Width = new GridLength(_sidebarWidth);
        }
        else
        {
            if (column.ActualWidth > 0) _sidebarWidth = column.ActualWidth;
            column.MinWidth = 0;
            column.Width = new GridLength(0);
        }
    }

    private static IEnumerable<string> GetFiles(DragEventArgs e)
    {
        foreach (var item in e.DataTransfer.Items)
            if (item.Formats.Contains(DataFormat.File) && item.TryGetRaw(DataFormat.File) is IStorageItem storage &&
                storage.TryGetLocalPath() is string path &&
                (path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) || ONNXStudio.Core.Python.PythonModel.IsPythonModelFile(path)))
                yield return path;
    }
}
