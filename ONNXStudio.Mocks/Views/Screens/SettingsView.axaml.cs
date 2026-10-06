using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace ONNXStudio.Mocks.Views.Screens;

public partial class SettingsView : UserControl
{
    public SettingsView()
    {
        InitializeComponent();
    }
    
    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
