using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace ONNXStudio.Mocks.Views.Screens;

public partial class ApiSandboxView : UserControl
{
    public ApiSandboxView()
    {
        InitializeComponent();
    }
    
    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
