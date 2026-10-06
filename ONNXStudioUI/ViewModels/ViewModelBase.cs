using CommunityToolkit.Mvvm.ComponentModel;

namespace ONNXStudioUI.ViewModels;

public abstract partial class ViewModelBase : ObservableObject
{
    [ObservableProperty]
    private string _title = string.Empty;

    [ObservableProperty]
    private bool _isBusy;
}
