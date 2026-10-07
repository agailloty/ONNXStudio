using System;
using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// Dashboard of loaded models (S-03).
/// </summary>
public partial class DashboardViewModel : ViewModelBase
{
    private readonly MainWindowViewModel _shell;
    private readonly IModelRegistry _registry;
    private readonly IModelLoadCoordinator _loader;
    private readonly ONNXStudioUI.Services.IFilePickerService _picker;

    [ObservableProperty]
    private ObservableCollection<IModel> _filteredModels = new();

    [ObservableProperty]
    private string _searchText = string.Empty;

    [ObservableProperty]
    private string _selectedSort = "Recently loaded";

    public DashboardViewModel(MainWindowViewModel shell, IModelRegistry registry, IModelLoadCoordinator loader, ONNXStudioUI.Services.IFilePickerService picker)
    {
        _shell = shell;
        _registry = registry;
        _loader = loader;
        _picker = picker;
        Title = "Dashboard";
        Refresh();
    }

    public IReadOnlyList<string> SortOptions { get; } = new[] { "Recently loaded", "Name", "Size" };

    [RelayCommand]
    private async Task OpenModelsAsync() => await _loader.LoadManyAsync(await _picker.PickModelFilesAsync());

    partial void OnSearchTextChanged(string value) => Refresh();
    partial void OnSelectedSortChanged(string value) => Refresh();

    [RelayCommand]
    private void Refresh()
    {
        IEnumerable<IModel> models = _registry.Models;

        if (!string.IsNullOrWhiteSpace(SearchText))
        {
            models = models.Where(m =>
                m.Name.Contains(SearchText, StringComparison.OrdinalIgnoreCase) ||
                m.FilePath.Contains(SearchText, StringComparison.OrdinalIgnoreCase));
        }

        models = SelectedSort switch
        {
            "Name" => models.OrderBy(m => m.Name, StringComparer.OrdinalIgnoreCase),
            "Size" => models.OrderByDescending(m => m.FileSize),
            _ => models.OrderByDescending(m => m.LoadedAt)
        };

        FilteredModels = new ObservableCollection<IModel>(models);
    }

    [RelayCommand]
    private void ShowSettings()
    {
        _shell.ShowSettings();
    }

    [RelayCommand]
    private void InspectModel(IModel model)
    {
        _shell.ShowInspector(model);
    }

    [RelayCommand]
    private void RunModel(IModel model)
    {
        _shell.ShowPlayground(model);
    }

    [RelayCommand]
    private void ServeModel(IModel model)
    {
        _shell.ShowApiConfig(model);
    }

    [RelayCommand]
    private void UnloadModel(IModel model)
    {
        _registry.Unload(model.Id);
        _shell.ShowToast($"Model '{model.Name}' unloaded");
        Refresh();
    }
}
