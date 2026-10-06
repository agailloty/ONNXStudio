using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Mocks.Models;

namespace ONNXStudio.Mocks.ViewModels.Screens;

public partial class WelcomeViewModel : ViewModelBase
{
    private readonly MainViewModel _mainViewModel;
    
    [ObservableProperty]
    private string _dragDropText = "Or drag and drop an .onnx file here";
    
    [ObservableProperty]
    private bool _isDragOver;
    
    [ObservableProperty]
    private bool _isDragValid;
    
    [ObservableProperty]
    private bool _isLoading;
    
    [ObservableProperty]
    private string _statusMessage = string.Empty;
    
    public WelcomeViewModel(MainViewModel mainViewModel)
    {
        _mainViewModel = mainViewModel;
        Title = "Welcome to ONNX Studio";
    }
    
    [RelayCommand]
    private async Task OpenModelAsync()
    {
        // Simulate file picker - in real app, use AvaloniaUI.FileDialogs
        IsLoading = true;
        StatusMessage = "Opening file picker...";
        
        // Simulate delay
        await Task.Delay(500);
        
        // For mock purposes, create a sample model
        var mockModel = MockDataGenerator.CreateMockResNet50();
        mockModel.FilePath = "/mock/path/resnet50.onnx";
        
        _mainViewModel.AddModel(mockModel);
        _mainViewModel.ShowModelLoading(mockModel.FilePath);
        
        IsLoading = false;
    }
    
    [RelayCommand]
    private void HandleDragOver(bool isOver)
    {
        IsDragOver = isOver;
    }
    
    [RelayCommand]
    private void HandleDragDrop(string[] files)
    {
        if (files == null || files.Length == 0) return;
        
        foreach (var file in files)
        {
            if (Path.GetExtension(file).Equals(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                _mainViewModel.ShowModelLoading(file);
                break;
            }
        }
        
        IsDragOver = false;
        IsDragValid = false;
    }
    
    [RelayCommand]
    private void LearnMore()
    {
        // Open documentation URL
        StatusMessage = "Opening documentation...";
    }
    
    [RelayCommand]
    private void ViewExamples()
    {
        // Show example models
        var examples = new[]
        {
            MockDataGenerator.CreateMockResNet50(),
            MockDataGenerator.CreateMockBert(),
            MockDataGenerator.CreateMockMobileNet()
        };
        
        foreach (var example in examples)
        {
            _mainViewModel.AddModel(example);
        }
        
        _mainViewModel.ShowDashboard();
    }
    
    public override void OnLoaded()
    {
        base.OnLoaded();
        StatusMessage = "Ready to load ONNX models";
    }
}
