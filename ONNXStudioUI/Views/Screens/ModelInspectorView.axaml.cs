using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace ONNXStudioUI.Views.Screens;

public partial class ModelInspectorView : UserControl
{
    public ModelInspectorView()
    {
        InitializeComponent();
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
