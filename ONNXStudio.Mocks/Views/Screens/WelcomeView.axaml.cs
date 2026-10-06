using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace ONNXStudio.Mocks.Views.Screens;

public partial class WelcomeView : UserControl
{
    public WelcomeView()
    {
        InitializeComponent();
    }
    
    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
