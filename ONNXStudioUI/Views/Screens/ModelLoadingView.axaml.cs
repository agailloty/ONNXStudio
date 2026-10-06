using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace ONNXStudioUI.Views.Screens;

public partial class ModelLoadingView : UserControl
{
    public ModelLoadingView()
    {
        InitializeComponent();
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
