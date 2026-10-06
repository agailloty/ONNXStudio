using Avalonia.Controls;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

        // Wire the storage picker once the window (TopLevel) is created.
        if (App.Services.GetService(typeof(FilePickerService)) is FilePickerService filePicker)
        {
            filePicker.Initialize(() => StorageProvider);
        }
    }
}
