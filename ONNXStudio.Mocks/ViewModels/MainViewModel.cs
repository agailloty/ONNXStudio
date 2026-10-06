using System;
using System.Collections.ObjectModel;
using System.Linq;
using Avalonia;
using Avalonia.Styling;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Mocks.Models;
using ONNXStudio.Mocks.ViewModels.Screens;
using ThemeMode = ONNXStudio.Mocks.Models.ThemeMode;

namespace ONNXStudio.Mocks.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    [ObservableProperty]
    private ViewModelBase? _currentViewModel;
    
    [ObservableProperty]
    private ObservableCollection<OnnxModel> _models = new();
    
    [ObservableProperty]
    private OnnxModel? _selectedModel;
    
    [ObservableProperty]
    private AppSettings _settings = new();
    
    [ObservableProperty]
    private string _statusMessage = "Ready";
    
    public MainViewModel()
    {
        Title = "ONNX Studio";
        LoadSettings();
        ApplyTheme(Settings.Theme);

        // Initialize with mock models for demo
        if (Models.Count == 0)
        {
            // Add some mock models
            Models.Add(MockDataGenerator.CreateMockResNet50());
            Models.Add(MockDataGenerator.CreateMockBert());
            Models.Add(MockDataGenerator.CreateMockMobileNet());
            Models.Add(MockDataGenerator.CreateMockCaliforniaHousingPipeline());
        }
        
        // Start with dashboard if models exist, otherwise welcome
        if (Models.Count > 0)
        {
            CurrentViewModel = new DashboardViewModel(this);
        }
        else
        {
            CurrentViewModel = new WelcomeViewModel(this);
        }
    }
    
    public void NavigateTo(ViewModelBase viewModel)
    {
        CurrentViewModel = viewModel;
    }
    
    [RelayCommand]
    public void ShowWelcome()
    {
        CurrentViewModel = new WelcomeViewModel(this);
    }
    
    [RelayCommand]
    public void ShowDashboard()
    {
        CurrentViewModel = new DashboardViewModel(this);
    }
    
    [RelayCommand]
    public void ShowModelLoading(string filePath)
    {
        CurrentViewModel = new ModelLoadingViewModel(this, filePath);
    }
    
    [RelayCommand]
    public void ShowModelInspector(OnnxModel model)
    {
        SelectedModel = model;
        CurrentViewModel = new ModelInspectorViewModel(this, model);
    }
    
    [RelayCommand]
    public void ShowInferencePlayground(OnnxModel model)
    {
        SelectedModel = model;
        CurrentViewModel = new InferencePlaygroundViewModel(this, model);
    }
    
    [RelayCommand]
    public void ShowApiSandbox(OnnxModel model)
    {
        SelectedModel = model;
        CurrentViewModel = new ApiSandboxViewModel(this, model);
    }

    [RelayCommand]
    public void ShowApiConfig(OnnxModel model)
    {
        SelectedModel = model;
        CurrentViewModel = new ApiConfigViewModel(this, model);
    }
    
    [RelayCommand]
    public void ShowSettings()
    {
        CurrentViewModel = new SettingsViewModel(this, Settings);
    }
    
    public void AddModel(OnnxModel model)
    {
        Models.Add(model);
        SelectedModel = model;
        StatusMessage = $"Model '{model.Name}' loaded successfully";
        SaveSettings();
    }
    
    public void RemoveModel(OnnxModel model)
    {
        Models.Remove(model);
        if (SelectedModel == model)
        {
            SelectedModel = Models.FirstOrDefault();
        }
        StatusMessage = $"Model '{model.Name}' unloaded";
        SaveSettings();
    }
    
    [RelayCommand]
    public void BackToDashboard()
    {
        if (Models.Count > 0)
        {
            ShowDashboard();
        }
        else
        {
            ShowWelcome();
        }
    }
    
    private void LoadSettings()
    {
        // TODO: Load from config file
        // For now, use defaults
        Settings = new AppSettings();
    }

    /// <summary>
    /// Applies the given theme to the whole application (FluentTheme + custom palettes).
    /// </summary>
    public void ApplyTheme(ThemeMode theme)
    {
        Settings.Theme = theme;

        if (Application.Current is { } app)
        {
            app.RequestedThemeVariant = theme switch
            {
                ThemeMode.Dark => ThemeVariant.Dark,
                ThemeMode.Light => ThemeVariant.Light,
                _ => null // System
            };
        }
    }
    
    public void SaveSettings()
    {
        // TODO: Save to config file
        StatusMessage = "Settings saved";
    }
}
