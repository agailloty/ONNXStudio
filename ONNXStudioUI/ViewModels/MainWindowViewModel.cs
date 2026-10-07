using System;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using ONNXStudio.Api;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Services;

namespace ONNXStudioUI.ViewModels;

/// <summary>
/// Shell view model: navigation between screens, model registry projection,
/// status bar, toasts and theme. Screen view models are created through DI.
/// </summary>
public partial class MainWindowViewModel : ObservableObject, IDisposable
{
    private readonly IServiceProvider _services;
    private readonly IModelRegistry _registry;
    private readonly IGraphAnalysisService _graphService;
    private readonly IToastService _toast;
    private readonly IThemeService _theme;
    private readonly ILogger<MainWindowViewModel> _logger;
    private readonly Dictionary<string, ViewModelBase> _screenCache = new();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsDashboardActive))]
    [NotifyPropertyChangedFor(nameof(IsSettingsActive))]
    private ViewModelBase? _currentViewModel;

    [ObservableProperty]
    private bool _isSidebarVisible = true;

    [ObservableProperty]
    private string _statusDetails = string.Empty;

    [ObservableProperty]
    private ExplorerNode? _selectedExplorerNode;

    private readonly ApiServerHost _api;
    private bool _syncingExplorer;

    /// <summary>Screens currently open in the editor area.</summary>
    public ObservableCollection<EditorTab> Tabs { get; } = new();

    /// <summary>Loaded models and their screens, shown in the side bar.</summary>
    public ObservableCollection<ExplorerNode> ExplorerNodes { get; } = new();

    public bool IsDashboardActive => CurrentViewModel is ViewModels.Screens.DashboardViewModel;

    public bool IsSettingsActive => CurrentViewModel is ViewModels.Screens.SettingsViewModel;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasModels))]
    private ObservableCollection<OnnxModel> _models = new();

    [ObservableProperty]
    private string _statusMessage = "Ready";

    [ObservableProperty]
    private string? _toastMessage;

    public MainWindowViewModel(
        IServiceProvider services,
        IModelRegistry registry,
        IGraphAnalysisService graphService,
        IToastService toast,
        IThemeService theme,
        ILogger<MainWindowViewModel> logger)
    {
        _services = services;
        _registry = registry;
        _graphService = graphService;
        _toast = toast;
        _theme = theme;
        _logger = logger;

        _api = services.GetRequiredService<ApiServerHost>();
        _api.StateChanged += OnApiStateChanged;
        _registry.ModelAdded += OnModelAdded;
        _registry.ModelRemoved += OnModelRemoved;
        _toast.ToastChanged += OnToastChanged;

        RefreshModels();
        UpdateStatusDetails();
        // NOTE: navigation is triggered by the App after this singleton is fully
        // constructed (screen view models depend on this shell view model).
    }

    public string AppVersion => "1.0.0";

    // ----- navigation -----

    [RelayCommand]
    public void ShowWelcome()
    {
        CurrentViewModel = _services.GetService(typeof(ViewModels.Screens.WelcomeViewModel)) is ViewModels.Screens.WelcomeViewModel welcome
            ? welcome
            : null;
        StatusMessage = Models.Count == 0 ? "No model loaded" : $"{Models.Count} model(s) loaded";
    }

    [RelayCommand]
    public void ShowDashboard()
    {
        CurrentViewModel = _services.GetService(typeof(ViewModels.Screens.DashboardViewModel)) is ViewModels.Screens.DashboardViewModel dashboard
            ? dashboard
            : null;
    }

    public void ShowSettings()
    {
        CurrentViewModel = new ViewModels.Screens.SettingsViewModel(
            this,
            _services.GetRequiredService<IThemeService>(),
            _services.GetRequiredService<IToastService>(),
            _services.GetRequiredService<ONNXStudio.Api.ApiServerHost>(),
            _services.GetRequiredService<SettingsStore>());
    }

    /// <summary>
    /// Sets the current screen directly (used by the model load coordinator).
    /// </summary>
    public void SetCurrentScreen(ViewModelBase viewModel)
    {
        CurrentViewModel = viewModel;
    }

    public bool HasModels => Models.Count > 0;

    [RelayCommand]
    private void OpenSettings() => ShowSettings();

    [RelayCommand]
    private void ToggleSidebar() => IsSidebarVisible = !IsSidebarVisible;

    [RelayCommand]
    private async Task OpenModelsAsync()
    {
        var picker = _services.GetRequiredService<IFilePickerService>();
        var loader = _services.GetRequiredService<IModelLoadCoordinator>();
        await loader.LoadManyAsync(await picker.PickModelFilesAsync());
    }

    [RelayCommand]
    private void UnloadModel(OnnxModel? model)
    {
        if (model == null) return;
        _registry.Unload(model.Id);
        ShowToast($"Model '{model.Name}' unloaded");
    }

    // ----- editor tabs -----

    [RelayCommand]
    private void ActivateTab(EditorTab? tab)
    {
        if (tab == null || tab.IsActive) return;
        Navigate(tab.Key);
    }

    [RelayCommand]
    private void CloseTab(EditorTab? tab)
    {
        if (tab == null) return;
        var index = Tabs.IndexOf(tab);
        if (index < 0) return;
        var wasActive = tab.IsActive;
        Tabs.RemoveAt(index);
        if (!wasActive) return;

        if (Tabs.Count == 0) ShowWelcome();
        else Navigate(Tabs[Math.Min(index, Tabs.Count - 1)].Key);
    }

    [RelayCommand]
    private void CloseActiveTab() => CloseTab(Tabs.FirstOrDefault(t => t.IsActive));

    private void Navigate(string key)
    {
        switch (key)
        {
            case "dashboard": ShowDashboard(); break;
            case "settings": ShowSettings(); break;
            default:
                if (_screenCache.TryGetValue(key, out var screen)) CurrentViewModel = screen;
                break;
        }
    }

    private string? KeyOf(ViewModelBase? screen) => screen switch
    {
        null => null,
        ViewModels.Screens.DashboardViewModel => "dashboard",
        ViewModels.Screens.SettingsViewModel => "settings",
        _ => _screenCache.FirstOrDefault(entry => ReferenceEquals(entry.Value, screen)).Key
    };

    private static string IconFor(string key) => key.Split(':')[0] switch
    {
        "dashboard" => "IconDashboard",
        "settings" => "IconSettings",
        "inspector" => "IconGraph",
        "playground" => "IconPlay",
        "apiconfig" => "IconApi",
        "sandbox" => "IconSend",
        _ => "IconBox"
    };

    partial void OnCurrentViewModelChanged(ViewModelBase? value)
    {
        var key = KeyOf(value);
        foreach (var tab in Tabs) tab.IsActive = tab.Key == key;

        if (key != null && value != null)
        {
            var tab = Tabs.FirstOrDefault(t => t.Key == key);
            if (tab == null)
            {
                tab = new EditorTab(key, value.Title, IconFor(key));
                Tabs.Add(tab);
            }
            tab.Title = value.Title;
            tab.IsActive = true;
        }

        SyncExplorerSelection(key);
    }

    // ----- model explorer -----

    partial void OnSelectedExplorerNodeChanged(ExplorerNode? value)
    {
        if (_syncingExplorer || value == null) return;
        var model = value.Model;
        switch (value.Key.Split(':')[0])
        {
            case "playground": ShowPlayground(model); break;
            case "apiconfig": ShowApiConfig(model); break;
            case "sandbox": ShowApiSandbox(model); break;
            default: ShowInspector(model); break;
        }
    }

    private void SyncExplorerSelection(string? key)
    {
        _syncingExplorer = true;
        try
        {
            var selected = key == null
                ? null
                : ExplorerNodes.SelectMany(n => n.Children.Prepend(n)).FirstOrDefault(n => n.Key == key);
            if (selected != null && ExplorerNodes.FirstOrDefault(n => n.Model.Id == selected.Model.Id) is { } parent) parent.IsExpanded = true;
            SelectedExplorerNode = selected;
        }
        finally { _syncingExplorer = false; }
    }

    private static ExplorerNode BuildExplorerNode(OnnxModel model)
    {
        var node = new ExplorerNode("model:" + model.Id, model.Name, "IconBox", model, model.FileSizeDisplay) { IsExpanded = false };
        node.Children.Add(new ExplorerNode("inspector:" + model.Id, "Inspector", "IconGraph", model));
        node.Children.Add(new ExplorerNode("playground:" + model.Id, "Inference", "IconPlay", model));
        node.Children.Add(new ExplorerNode("apiconfig:" + model.Id, "API", "IconApi", model));
        node.Children.Add(new ExplorerNode("sandbox:" + model.Id, "Sandbox", "IconSend", model));
        return node;
    }

    private void OnApiStateChanged()
    {
        Avalonia.Threading.Dispatcher.UIThread.Post(UpdateStatusDetails);
    }

    private void UpdateStatusDetails()
    {
        var models = Models.Count == 1 ? "1 model" : $"{Models.Count} models";
        var api = _api.IsRunning ? $"API localhost:{_api.Port}" : "API stopped";
        StatusDetails = $"{models}   {api}";
    }

    // ----- model screens (cached per model so state is preserved) -----

    public void ShowInspector(OnnxModel model)
    {
        CurrentViewModel = GetOrCreateScreen("inspector:" + model.Id,
            () => new ViewModels.Screens.ModelInspectorViewModel(this, _graphService, model));
    }

    // Playground navigation is wired in T9; API screens in T10.
    public void ShowPlayground(OnnxModel model)
    {
        CurrentViewModel = GetOrCreateScreen("playground:" + model.Id,
            () => new ViewModels.Screens.InferencePlaygroundViewModel(
                this,
                _services.GetRequiredService<IInferenceService>(),
                _services.GetRequiredService<IFormGenerationService>(),
                _services.GetRequiredService<IFilePickerService>(),
                model));
    }

    public void ShowApiConfig(OnnxModel model)
    {
        CurrentViewModel = GetOrCreateScreen("apiconfig:" + model.Id,
            () => new ViewModels.Screens.ApiConfigViewModel(
                this,
                _services.GetRequiredService<ONNXStudio.Api.ApiServerHost>(),
                _services.GetRequiredService<IToastService>(),
                model));
    }

    public void ShowApiSandbox(OnnxModel model)
    {
        CurrentViewModel = GetOrCreateScreen("sandbox:" + model.Id,
            () => new ViewModels.Screens.ApiSandboxViewModel(
                this,
                _services.GetRequiredService<ONNXStudio.Api.ApiServerHost>(),
                _services.GetRequiredService<IToastService>(),
                model));
    }

    private ViewModelBase GetOrCreateScreen(string key, Func<ViewModelBase> factory)
    {
        if (_screenCache.TryGetValue(key, out var cached))
        {
            return cached;
        }
        var vm = factory();
        _screenCache[key] = vm;
        return vm;
    }

    // ----- model registry projection -----

    private void OnModelAdded(object? sender, OnnxModel model)
    {
        Avalonia.Threading.Dispatcher.UIThread.Post(() =>
        {
            Models.Add(model);
            ExplorerNodes.Add(BuildExplorerNode(model));
            OnPropertyChanged(nameof(HasModels));
            UpdateStatusDetails();
            StatusMessage = $"Model '{model.Name}' loaded";
        });
    }

    private void OnModelRemoved(object? sender, OnnxModel model)
    {
        Avalonia.Threading.Dispatcher.UIThread.Post(() =>
        {
            Models.Remove(model);
            if (ExplorerNodes.FirstOrDefault(n => n.Model.Id == model.Id) is { } node) ExplorerNodes.Remove(node);
            OnPropertyChanged(nameof(HasModels));
            UpdateStatusDetails();
            foreach (var key in _screenCache.Keys.Where(k => k.EndsWith(":" + model.Id, StringComparison.Ordinal)).ToArray())
            {
                if (_screenCache.Remove(key, out var screen))
                {
                    if (Tabs.FirstOrDefault(t => t.Key == key) is { } tab) Tabs.Remove(tab);
                    if (screen is IDisposable disposable) disposable.Dispose();
                    if (ReferenceEquals(CurrentViewModel, screen)) ShowDashboard();
                }
            }
            StatusMessage = $"Model '{model.Name}' unloaded";
        });
    }

    private void RefreshModels()
    {
        Models = new ObservableCollection<OnnxModel>(_registry.Models);
        ExplorerNodes.Clear();
        foreach (var model in Models) ExplorerNodes.Add(BuildExplorerNode(model));
    }

    // ----- toasts -----

    private void OnToastChanged()
    {
        Avalonia.Threading.Dispatcher.UIThread.Post(() => ToastMessage = _toast.Current);
    }

    public void ShowToast(string message) => _toast.Show(message);

    public void Dispose()
    {
        _registry.ModelAdded -= OnModelAdded;
        _registry.ModelRemoved -= OnModelRemoved;
        _toast.ToastChanged -= OnToastChanged;
        _api.StateChanged -= OnApiStateChanged;
        foreach (var screen in _screenCache.Values.OfType<IDisposable>()) screen.Dispose();
        _screenCache.Clear();
    }
}
