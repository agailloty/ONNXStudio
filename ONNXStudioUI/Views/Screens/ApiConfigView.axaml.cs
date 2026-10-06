using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace ONNXStudioUI.Views.Screens;

public partial class ApiConfigView : UserControl
{
    public ApiConfigView()
    {
        InitializeComponent();
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
