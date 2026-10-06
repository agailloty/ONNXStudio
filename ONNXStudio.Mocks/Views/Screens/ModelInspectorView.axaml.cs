using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using ONNXStudio.Mocks.ViewModels.Screens;

namespace ONNXStudio.Mocks.Views.Screens;

public partial class ModelInspectorView : UserControl
{
    public ModelInspectorView()
    {
        InitializeComponent();
    }

    private void OnSearchKeyUp(object? sender, KeyEventArgs e)
    {
        if (DataContext is ModelInspectorViewModel vm)
        {
            vm.SearchNodesCommand.Execute(null);
        }
    }

    private void OnFilterTypeChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (DataContext is ModelInspectorViewModel vm)
        {
            vm.FilterNodesCommand.Execute(null);
        }
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
