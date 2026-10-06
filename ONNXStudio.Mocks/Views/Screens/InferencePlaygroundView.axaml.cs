using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace ONNXStudio.Mocks.Views.Screens;

public partial class InferencePlaygroundView : UserControl
{
    public InferencePlaygroundView()
    {
        InitializeComponent();
    }
    
    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
