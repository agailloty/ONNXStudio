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

    [ObservableProperty]
    private ObservableCollection<OnnxModel> _filteredModels = new();

    [ObservableProperty]
    private string _searchText = string.Empty;

    [ObservableProperty]
    private string _selectedSort = "Recently loaded";

    public DashboardViewModel(MainWindowViewModel shell, IModelRegistry registry)
    {
        _shell = shell;
        _registry = registry;
        Title = "Dashboard";
        Refresh();
    }

    public IReadOnlyList<string> SortOptions { get; } = new[] { "Recently loaded", "Name", "Size" };

    partial void OnSearchTextChanged(string value) => Refresh();
    partial void OnSelectedSortChanged(string value) => Refresh();

    [RelayCommand]
    private void Refresh()
    {
        IEnumerable<OnnxModel> models = _registry.Models;

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

        FilteredModels = new ObservableCollection<OnnxModel>(models);
    }

    [RelayCommand]
    private void InspectModel(OnnxModel model)
    {
        _shell.ShowInspector(model);
    }

    [RelayCommand]
    private void UnloadModel(OnnxModel model)
    {
        _registry.Unload(model.Id);
        _shell.ShowToast($"Model '{model.Name}' unloaded");
        Refresh();
    }
}
