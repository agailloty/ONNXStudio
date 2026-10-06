using System;
using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Mocks.Models;

namespace ONNXStudio.Mocks.ViewModels.Screens;

public partial class DashboardViewModel : ViewModelBase
{
    private readonly MainViewModel _mainViewModel;
    
    [ObservableProperty]
    private string _searchText = string.Empty;
    
    [ObservableProperty]
    private ObservableCollection<OnnxModel> _filteredModels = new();
    
    [ObservableProperty]
    private bool _isSelectAll = false;
    
    public DashboardViewModel(MainViewModel mainViewModel)
    {
        _mainViewModel = mainViewModel;
        Title = "ONNX Studio - Dashboard";
        
        // Initialize filtered models
        UpdateFilteredModels();
        
        // Subscribe to models collection changes
        _mainViewModel.Models.CollectionChanged += (sender, e) => UpdateFilteredModels();
    }
    
    private void UpdateFilteredModels()
    {
        if (string.IsNullOrWhiteSpace(SearchText))
        {
            FilteredModels = new ObservableCollection<OnnxModel>(_mainViewModel.Models);
        }
        else
        {
            FilteredModels = new ObservableCollection<OnnxModel>(
                _mainViewModel.Models.Where(m => 
                    m.Name.Contains(SearchText, StringComparison.OrdinalIgnoreCase) ||
                    m.FileName.Contains(SearchText, StringComparison.OrdinalIgnoreCase))
            );
        }
    }
    
    [RelayCommand]
    private void OpenModel()
    {
        _mainViewModel.ShowWelcome();
        // In real app, this would open file picker
        // For mock, we'll simulate adding a model
        var mockModel = MockDataGenerator.CreateMockSimpleModel();
        _mainViewModel.AddModel(mockModel);
    }
    
    [RelayCommand]
    private void ShowSettings()
    {
        _mainViewModel.ShowSettings();
    }
    
    [RelayCommand]
    private void Search()
    {
        UpdateFilteredModels();
    }
    
    [RelayCommand]
    private void InspectModel(OnnxModel model)
    {
        _mainViewModel.ShowModelInspector(model);
    }
    
    [RelayCommand]
    private void RunInference(OnnxModel model)
    {
        _mainViewModel.ShowInferencePlayground(model);
    }
    
    [RelayCommand]
    private void ShowApiConfig(OnnxModel model)
    {
        _mainViewModel.ShowApiConfig(model);
    }
    
    [RelayCommand]
    private void ToggleSelectAll()
    {
        IsSelectAll = !IsSelectAll;
        foreach (var model in FilteredModels)
        {
            // In real app, toggle selection
        }
    }
    
    [RelayCommand]
    private void UnloadSelected()
    {
        // In real app, unload selected models
        // For mock, unload last model
        if (_mainViewModel.Models.Count > 0)
        {
            var modelToRemove = _mainViewModel.Models[^1];
            _mainViewModel.RemoveModel(modelToRemove);
        }
    }
    
    [RelayCommand]
    private void UnloadAll()
    {
        foreach (var model in _mainViewModel.Models.ToList())
        {
            _mainViewModel.RemoveModel(model);
        }
    }
    
    public override void OnLoaded()
    {
        base.OnLoaded();
        UpdateFilteredModels();
    }
}
