using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace ONNXStudio.Mocks.Views.Screens;

public partial class DashboardView : UserControl
{
    public DashboardView()
    {
        InitializeComponent();
    }
    
    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
