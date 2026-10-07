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

        // Wire the storage picker once the window (TopLevel) is created.
        if (App.Services.GetService(typeof(FilePickerService)) is FilePickerService filePicker)
        {
            filePicker.Initialize(() => StorageProvider);
        }
    }

    private static IEnumerable<string> GetFiles(DragEventArgs e)
    {
        foreach (var item in e.DataTransfer.Items)
            if (item.Formats.Contains(DataFormat.File) && item.TryGetRaw(DataFormat.File) is IStorageItem storage &&
                storage.TryGetLocalPath() is string path && path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
                yield return path;
    }
}
