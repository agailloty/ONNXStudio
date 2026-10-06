using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using CommunityToolkit.Mvvm.ComponentModel;

namespace ONNXStudio.Mocks.ViewModels;

/// <summary>
/// Base ViewModel class with INotifyPropertyChanged
/// </summary>
public partial class ViewModelBase : ObservableObject
{
    [ObservableProperty]
    private string _title = string.Empty;
    
    [ObservableProperty]
    private bool _isBusy;
    
    [ObservableProperty]
    private string? _errorMessage;
    
    public virtual void OnLoaded() { }
    public virtual void OnUnloaded() { }
}
