using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Microsoft.Extensions.DependencyInjection;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Python;
using ONNXStudio.Core.Services;
using ONNXStudioUI.Controls;
using Avalonia.VisualTree;
using ONNXStudioUI.Services;
using ONNXStudioUI.ViewModels;
using ONNXStudioUI.ViewModels.Screens;
using ONNXStudioUI.Views.Screens;
using Xunit;

namespace ONNXStudio.UI.Tests;

internal sealed class FakePythonRuntime : IPythonRuntimeService
{
    public PythonInterpreter? Active { get; set; }
    public PythonSelection SelectionValue { get; set; } = PythonSelection.Automatic;
    public List<PythonInterpreter> Found { get; } = new();
    public List<PythonSelection> Selected { get; } = new();
    public Result<PythonInterpreter, PythonError>? Probe { get; set; }
    public bool ManagedInstalled { get; set; }

    public event Action? Changed;
    public PythonSelection Selection => SelectionValue;
    public bool IsManagedInstalled => ManagedInstalled;

    public void Select(PythonSelection selection)
    {
        SelectionValue = selection;
        Selected.Add(selection);
        Changed?.Invoke();
    }

    public Task<IReadOnlyList<PythonInterpreter>> DiscoverAsync(CancellationToken cancellationToken = default) => Task.FromResult<IReadOnlyList<PythonInterpreter>>(Found);

    public Task<Result<PythonInterpreter, PythonError>> ProbeCustomAsync(string pathOrFolder, CancellationToken cancellationToken = default)
        => Task.FromResult(Probe ?? Result<PythonInterpreter, PythonError>.Failure(new PythonError(PythonErrorCode.InvalidInterpreter, "No Python interpreter was found at this location.")));

    public Task<PythonInterpreter?> GetActiveAsync(bool refresh = false, CancellationToken cancellationToken = default) => Task.FromResult(Active);

    public Task<Result<PythonInterpreter, PythonError>> InstallPythonAsync(IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default)
    {
        progress?.Report(new PythonInstallProgress("download", "Downloading Python 3.12.15... 50%", 0.5));
        ManagedInstalled = true;
        return Task.FromResult(Result<PythonInterpreter, PythonError>.Success(Interpreter(PythonSource.Managed)));
    }

    public Task<Result<PythonInterpreter, PythonError>> InstallPackagesAsync(IProgress<PythonInstallProgress>? progress, CancellationToken cancellationToken = default)
    {
        progress?.Report(new PythonInstallProgress("pip", "Collecting scikit-learn"));
        Active = Interpreter(PythonSource.Managed, PythonRequirements.Managed.ToArray());
        return Task.FromResult(Result<PythonInterpreter, PythonError>.Success(Active));
    }

    public Task RemoveManagedAsync(CancellationToken cancellationToken = default)
    {
        ManagedInstalled = false;
        Active = null;
        return Task.CompletedTask;
    }

    public static PythonInterpreter Interpreter(PythonSource source, params string[] packages) => new("python", "3.12.4", "AMD64", source,
        PythonRequirements.Probed.ToDictionary(p => p, p => packages.Contains(p) ? "1.0.0" : null, StringComparer.OrdinalIgnoreCase));
}

internal sealed class FakePythonModelService : IPythonModelService
{
    public PythonModelInfo Info { get; set; } = new()
    {
        ClassName = "RandomForestClassifier",
        Module = "sklearn.ensemble",
        FinalEstimator = "RandomForestClassifier",
        IsClassifier = true,
        FeatureCount = 4,
        Classes = new[] { "0", "1", "2" },
        Methods = new[] { "predict", "predict_proba" },
        Parameters = new Dictionary<string, string> { ["n_estimators"] = "10" },
        TrainedWithSklearn = "1.4.0",
        RuntimeSklearn = "1.9.1",
        Warnings = new[] { "InconsistentVersionWarning: ..." }
    };

    public PythonError? Failure { get; set; }
    public PythonPredictionRequest? LastPrediction { get; private set; }
    public PythonConversionRequest? LastConversion { get; private set; }
    public bool? LastInspectTrust { get; private set; }

    public Task<Result<PythonModelInfo, PythonError>> InspectAsync(string modelPath, bool trustConfirmed, CancellationToken cancellationToken = default)
    {
        LastInspectTrust = trustConfirmed;
        return Task.FromResult(Failure != null
            ? Result<PythonModelInfo, PythonError>.Failure(Failure)
            : Result<PythonModelInfo, PythonError>.Success(Info));
    }

    public Task<Result<PythonPredictionResult, PythonError>> PredictAsync(PythonPredictionRequest request, CancellationToken cancellationToken = default)
    {
        LastPrediction = request;
        if (Failure != null) return Task.FromResult(Result<PythonPredictionResult, PythonError>.Failure(Failure));
        var output = new PythonPredictionOutput("predict", "int64", new[] { request.Rows.Count }, false, request.Rows.Select(_ => "1").ToArray(), System.Text.Json.Nodes.JsonNode.Parse("[" + string.Join(",", request.Rows.Select(_ => "1")) + "]"));
        return Task.FromResult(Result<PythonPredictionResult, PythonError>.Success(
            new PythonPredictionResult(new[] { output }, new[] { "0", "1" }, request.Rows.Count, 5, Array.Empty<string>())));
    }

    public Task<Result<PythonConversionResult, PythonError>> ConvertAsync(PythonConversionRequest request, CancellationToken cancellationToken = default)
    {
        LastConversion = request;
        if (Failure != null) return Task.FromResult(Result<PythonConversionResult, PythonError>.Failure(Failure));
        return Task.FromResult(Result<PythonConversionResult, PythonError>.Success(new PythonConversionResult
        {
            OutputPath = request.OutputPath,
            TargetOpset = 17,
            RequestedOpset = request.TargetOpset,
            Size = 2048,
            Inputs = new[] { new PythonOnnxTensor("float_input", 1, new long?[] { null, 4 }) },
            Outputs = new[] { new PythonOnnxTensor("label", 7, new long?[] { null }) },
            Validation = new PythonConversionValidation(true, true, "100.0% of the outputs match.")
        }));
    }
}

internal sealed class RecordingCoordinator : IModelLoadCoordinator
{
    public List<string> Loaded { get; } = new();
    public Task RegisterAsync(IModel model) => Task.CompletedTask;
    public Task LoadAsync(string filePath) { Loaded.Add(filePath); return Task.CompletedTask; }
    public Task LoadManyAsync(IEnumerable<string> filePaths) { Loaded.AddRange(filePaths); return Task.CompletedTask; }
}

public class PythonScreensTests
{
    private static string TempModel(string name = "model.joblib")
    {
        var directory = Path.Combine(Path.GetTempPath(), "onnxstudio-ui-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, name);
        File.WriteAllText(path, "x");
        return path;
    }

    private static PythonModelViewModel CreateModelScreen(
        ServiceProvider services, FakePythonModelService model, FakePythonRuntime runtime, RecordingCoordinator? coordinator = null, string? file = null)
    {
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var registered = services.GetRequiredService<IPythonModelRegistry>().Register(file ?? TempModel());
        return new PythonModelViewModel(shell, model, runtime, services.GetRequiredService<IFilePickerService>(),
            services.GetRequiredService<IToastService>(), () => coordinator, registered);
    }

    [AvaloniaFact]
    public async Task OpeningJoblibAndPickleFilesShowsAPythonScreenInsteadOfTheOnnxLoader()
    {
        await using var services = UiTestSetup.Services();
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var file = TempModel("forest.joblib");

        await services.GetRequiredService<IModelLoadCoordinator>().LoadManyAsync(new[] { file, TempModel("other.pkl") });
        Dispatcher.UIThread.RunJobs();

        Assert.True(shell.HasPythonModels);
        Assert.Equal(2, shell.PythonModels.Count);
        Assert.Empty(shell.Models);
        var screen = Assert.IsType<PythonModelViewModel>(shell.CurrentViewModel);
        Assert.Equal("other.pkl", screen.Model.Name);
        Assert.Equal("pymodel:" + screen.Model.Id, Assert.Single(shell.Tabs, t => t.IsActive).Key);
        Assert.Equal(screen.Model, shell.SelectedPythonModel);

        // The same file is not registered twice.
        shell.OpenPythonModel(file);
        Dispatcher.UIThread.RunJobs();
        Assert.Equal(2, shell.PythonModels.Count);
        Assert.Equal("forest.joblib", Assert.IsType<PythonModelViewModel>(shell.CurrentViewModel).Model.Name);

        shell.UnloadPythonModelCommand.Execute(shell.PythonModels[0]);
        Dispatcher.UIThread.RunJobs();
        Assert.Single(shell.PythonModels);
        Assert.IsType<DashboardViewModel>(shell.CurrentViewModel);
        Assert.DoesNotContain(shell.Tabs, t => t.Title == "forest.joblib");
    }

    [AvaloniaFact]
    public async Task MissingPythonFileIsReportedWithoutOpeningAScreen()
    {
        await using var services = UiTestSetup.Services();
        var shell = services.GetRequiredService<MainWindowViewModel>();

        shell.OpenPythonModel(Path.Combine(Path.GetTempPath(), "does-not-exist.joblib"));

        Assert.False(shell.HasPythonModels);
        Assert.Contains("does not exist", services.GetRequiredService<IToastService>().Current);
    }

    [AvaloniaFact]
    public async Task ModelCannotBeLoadedUntilTheUserConfirmsTrust()
    {
        await using var services = UiTestSetup.Services();
        var service = new FakePythonModelService();
        var runtime = new FakePythonRuntime { Active = FakePythonRuntime.Interpreter(PythonSource.Managed, PythonRequirements.Managed.ToArray()) };
        var screen = CreateModelScreen(services, service, runtime);
        await screen.RefreshRuntimeAsync();

        Assert.True(screen.RuntimeReady);
        Assert.False(screen.CanLoad);
        Assert.False(screen.IsLoaded);

        await screen.LoadAsync();
        Assert.Null(service.LastInspectTrust);

        screen.IsTrusted = true;
        Assert.True(screen.CanLoad);
        await screen.LoadAsync();

        Assert.True(service.LastInspectTrust);
        Assert.True(screen.IsLoaded);
        Assert.True(screen.CanConvert);
        Assert.Equal("RandomForestClassifier", screen.SummaryText);
        Assert.Equal("4", screen.FeaturesText);
        Assert.Equal("0, 1, 2", screen.ClassesText);
        Assert.Contains("trained with scikit-learn 1.4.0, loaded with 1.9.1", screen.VersionText);
        Assert.Single(screen.Warnings);
        var inspector = Assert.IsType<ModelInspectorViewModel>(services.GetRequiredService<MainWindowViewModel>().CurrentViewModel);
        Assert.Same(screen.LoadedModel, inspector.Model);
        var root = Assert.Single(inspector.Components).Children[1];
        Assert.Equal("10", root.Parameters.Single(p => p.Key == "n_estimators").Value);
        Assert.Contains(root.Parameters, p => p.Key == "Trained with scikit-learn" && p.Value == "1.4.0");
    }

    [AvaloniaFact]
    public async Task LoadFailuresAreShownAndKeepTheModelUnloaded()
    {
        await using var services = UiTestSetup.Services();
        var service = new FakePythonModelService { Failure = new PythonError(PythonErrorCode.LoadFailed, "The file could not be loaded.", "Traceback\nValueError: bad pickle") };
        var screen = CreateModelScreen(services, service, new FakePythonRuntime());
        screen.IsTrusted = true;

        await screen.LoadAsync();

        Assert.False(screen.IsLoaded);
        Assert.Contains("could not be loaded", screen.LoadError);
        Assert.Contains("ValueError: bad pickle", screen.LoadError);
    }

    [AvaloniaFact]
    public async Task LoadedSklearnUsesTheSharedGraphStructureAndNavigation()
    {
        var service = new FakePythonModelService
        {
            Info = new PythonModelInfo
            {
                ClassName = "Pipeline", IsPipeline = true, FeatureCount = 2,
                Steps = [new("scale_", "StandardScaler"), new("classifier", "LogisticRegression")],
                Components = new PythonComponent
                {
                    Name = "pipeline", ClassName = "Pipeline", Kind = "Pipeline",
                    Children =
                    [
                        new() { Name = "scale_", ClassName = "StandardScaler", Kind = "Transformer" },
                        new()
                        {
                            Name = "classifier", ClassName = "LogisticRegression", Kind = "Classifier",
                            Parameters = [new("C", "1.0")],
                            Fitted = [new() { Name = "coef_", Kind = "array", Shape = [1, 2], Values = ["0.5", "1.5"], Count = 2 }]
                        }
                    ]
                }
            }
        };
        await using var services = UiTestSetup.Services(configure: s =>
        {
            s.AddSingleton<IPythonModelService>(service);
            s.AddSingleton<IPythonRuntimeService>(new FakePythonRuntime());
        });
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var file = TempModel();
        shell.OpenPythonModel(file);
        var import = Assert.IsType<PythonModelViewModel>(shell.CurrentViewModel);
        import.IsTrusted = true;
        await import.LoadAsync();

        var inspector = Assert.IsType<ModelInspectorViewModel>(shell.CurrentViewModel);
        Assert.Same(import.LoadedModel, Assert.Single(shell.Models));
        Assert.False(shell.HasPythonModels);
        Assert.Same(inspector.Model, services.GetRequiredService<IModelRegistry>().GetById(inspector.Model.Id));
        Assert.Equal("inspector:" + inspector.Model.Id, shell.SelectedExplorerNode!.Key);
        Assert.Equal(2, inspector.Nodes.Count);
        var edge = Assert.Single(inspector.Model.Graph.Edges);
        Assert.Equal(inspector.Nodes[0].Node.Id, edge.FromNodeId);
        Assert.Equal(inspector.Nodes[1].Node.Id, edge.ToNodeId);

        var view = new ModelInspectorView { DataContext = inspector };
        var window = new Window { Width = 1200, Height = 800, Content = view };
        try
        {
            window.Show();
            window.UpdateLayout();
            var graph = Assert.Single(view.GetVisualDescendants().OfType<GraphViewer>());
            Assert.Same(inspector.Model.Graph, graph.Graph);
            inspector.ShowStructureViewCommand.Execute(null);
            var classifier = Assert.Single(inspector.Components).Children[1].Children[1];
            inspector.SelectedComponent = classifier;
            window.UpdateLayout();
            Assert.Equal("LogisticRegression", inspector.SelectedNode!.OpType);
            Assert.Equal("1.0", classifier.Parameters.Single(p => p.Key == "C").Value);
            Assert.Equal("[0.5, 1.5]", Assert.Single(classifier.Fitted).Value);
            Assert.False(graph.IsEffectivelyVisible);
            inspector.ShowGraphViewCommand.Execute(null);
            window.UpdateLayout();
            Assert.True(graph.IsEffectivelyVisible);
            Assert.Same(inspector.SelectedNode, graph.SelectedNode);
        }
        finally { window.Close(); }

        foreach (var (target, expected) in new (string, Type)[]
                 { ("playground", typeof(InferencePlaygroundViewModel)), ("api", typeof(ApiConfigViewModel)), ("sandbox", typeof(ApiSandboxViewModel)) })
        {
            import.OpenModelCommand.Execute(target);
            Assert.IsType(expected, shell.CurrentViewModel);
        }
        shell.OpenPythonModel(file);
        Assert.Same(inspector, shell.CurrentViewModel);
        Assert.Single(shell.Models);
        shell.UnloadModelCommand.Execute(inspector.Model);
        Assert.Empty(shell.Models);
        Assert.Empty(shell.ExplorerNodes);
        Assert.DoesNotContain(shell.Tabs, t => t.Key.EndsWith(":" + inspector.Model.Id));
    }

    [AvaloniaFact]
    public async Task InferenceUsesTheSharedPlaygroundAndValidatesItsTensorInputs()
    {
        var service = new FakePythonModelService
        {
            Info = new PythonModelInfo { ClassName = "Pipeline", FeatureNames = new[] { "a", "b" }, FeatureCount = 2, Methods = new[] { "predict" } }
        };
        await using var services = UiTestSetup.Services(configure: s => s.AddSingleton<IPythonModelService>(service));
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var screen = CreateModelScreen(services, service, new FakePythonRuntime());
        screen.IsTrusted = true;
        await screen.LoadAsync();

        screen.OpenModelCommand.Execute("playground");
        var playground = Assert.IsType<InferencePlaygroundViewModel>(shell.CurrentViewModel);
        var input = Assert.Single(playground.Fields);
        Assert.Equal("0, 0", input.ValueText);
        input.ValueText = "1,2,3,4";
        await playground.RunInferenceCommand.ExecuteAsync(null);

        Assert.Null(playground.Error);
        Assert.True(playground.HasResult);
        Assert.Equal(new[] { "a", "b" }, service.LastPrediction!.Columns);
        Assert.Equal(2, service.LastPrediction.Rows.Count);
        Assert.Equal(new[] { "1", "2" }, service.LastPrediction.Rows[0]);
        Assert.Equal(new[] { "3", "4" }, service.LastPrediction.Rows[1]);
        Assert.Equal("all", service.LastPrediction.Method);
        Assert.True(service.LastPrediction.TrustConfirmed);
        Assert.Equal(new[] { "1", "1" }, Assert.Single(playground.Outputs).DisplayValues);

        input.ValueText = "1,2,3";
        await playground.RunInferenceCommand.ExecuteAsync(null);
        Assert.NotNull(playground.Error);
        Assert.False(playground.HasResult);
        Assert.Empty(playground.Outputs);
    }

    [AvaloniaTheory]
    [InlineData("forest.joblib")]
    [InlineData("forest.pkl")]
    [InlineData("forest.pickle")]
    public async Task ExportCanBeReopenedFromExplorerAndInspectorAfterClosingItsTab(string filename)
    {
        var service = new FakePythonModelService();
        await using var services = UiTestSetup.Services(configure: s =>
        {
            s.AddSingleton<IPythonModelService>(service);
            s.AddSingleton<IPythonRuntimeService>(new FakePythonRuntime());
        });
        var shell = services.GetRequiredService<MainWindowViewModel>();
        shell.OpenPythonModel(TempModel(filename));
        var export = Assert.IsType<PythonModelViewModel>(shell.CurrentViewModel);
        export.IsTrusted = true;
        await export.LoadAsync();
        export.OpsetText = "17";
        export.LoadAfterConversion = false;
        var output = export.OutputPath;
        var inspector = Assert.IsType<ModelInspectorViewModel>(shell.CurrentViewModel);
        var exportNode = Assert.Single(Assert.Single(shell.ExplorerNodes).Children, n => n.Header == "Export to ONNX");
        var key = "pymodel:" + export.Model.Id;

        // Close the import tab after loading, then reopen it through the explorer.
        shell.CloseTabCommand.Execute(Assert.Single(shell.Tabs, t => t.Key == key));
        shell.SelectedExplorerNode = exportNode;
        Assert.Same(export, shell.CurrentViewModel);
        Assert.Same(exportNode, shell.SelectedExplorerNode);
        Assert.True(Assert.Single(shell.Tabs, t => t.Key == key).IsActive);
        Assert.True(export.CanConvert);
        Assert.Equal("17", export.OpsetText);
        Assert.Equal(output, export.OutputPath);

        // Closing the active export tab must also allow reopening from the visible inspector button.
        shell.CloseActiveTabCommand.Execute(null);
        Assert.Same(inspector, shell.CurrentViewModel);
        var view = new ModelInspectorView { DataContext = inspector };
        var window = new Window { Width = 1280, Height = 800, Content = view };
        try
        {
            window.Show();
            window.UpdateLayout();
            var button = Assert.Single(view.GetVisualDescendants().OfType<Button>(), b => Equals(b.Content, "Export to ONNX"));
            Assert.True(button.IsEffectivelyVisible);
            Assert.True(button.IsEnabled);
            Assert.NotNull(button.Command);
            button.Command.Execute(button.CommandParameter);
            Assert.Same(export, shell.CurrentViewModel);
            Assert.Same(exportNode, shell.SelectedExplorerNode);
            Assert.Single(shell.Tabs, t => t.Key == key);
            await export.ConvertAsync();
            Assert.Null(export.ConversionError);
            Assert.Equal(output, service.LastConversion!.OutputPath);
            Assert.Equal(17, service.LastConversion.TargetOpset);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task ConversionPassesOptionsAndOpensTheOnnxModel()
    {
        await using var services = UiTestSetup.Services();
        var service = new FakePythonModelService();
        var coordinator = new RecordingCoordinator();
        var file = TempModel("forest.joblib");
        var screen = CreateModelScreen(services, service, new FakePythonRuntime(), coordinator, file);
        screen.IsTrusted = true;
        await screen.LoadAsync();

        Assert.Equal(Path.ChangeExtension(file, ".onnx"), screen.OutputPath);

        screen.OpsetText = "17";
        screen.InputTypesText = "age:float, city:string";
        await screen.ConvertAsync();

        Assert.Null(screen.ConversionError);
        var request = service.LastConversion!;
        Assert.Equal(17, request.TargetOpset);
        Assert.True(request.DisableZipMap);
        Assert.True(request.Validate);
        Assert.True(request.TrustConfirmed);
        Assert.Equal(new[] { "age:float", "city:string" }, request.InputTypes);
        Assert.True(screen.HasConversionResult);
        Assert.Contains("opset 17", screen.ConversionSummary);
        Assert.Contains(screen.ConversionDetails, d => d.StartsWith("Validation: 100.0%"));
        Assert.Equal(new[] { Path.ChangeExtension(file, ".onnx") }, coordinator.Loaded);

        coordinator.Loaded.Clear();
        screen.LoadAfterConversion = false;
        await screen.ConvertAsync();
        Assert.Empty(coordinator.Loaded);
    }

    [AvaloniaFact]
    public async Task ConvertedModelOpensInTheInspectorAndEveryOnnxScreenIsReachable()
    {
        await using var services = UiTestSetup.Services();
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var output = Path.Combine(Path.GetDirectoryName(TempModel())!, "converted.onnx");
        File.Copy(Path.Combine(AppContext.BaseDirectory, "fixtures", "sklearn_pipeline.onnx"), output);
        var registered = services.GetRequiredService<IPythonModelRegistry>().Register(TempModel());
        var screen = new PythonModelViewModel(shell, new FakePythonModelService(), new FakePythonRuntime(),
            services.GetRequiredService<IFilePickerService>(), services.GetRequiredService<IToastService>(),
            () => services.GetRequiredService<IModelLoadCoordinator>(), registered);
        screen.IsTrusted = true;
        await screen.LoadAsync();
        screen.OutputPath = output;

        await screen.ConvertAsync();
        Dispatcher.UIThread.RunJobs();

        Assert.Null(screen.ConversionError);
        var inspector = Assert.IsType<ModelInspectorViewModel>(shell.CurrentViewModel);
        Assert.Equal(output, inspector.Model.FilePath);
        Assert.False(inspector.CanExportToOnnx);
        Assert.False(inspector.ExportToOnnxCommand.CanExecute(null));
        Assert.DoesNotContain(shell.ExplorerNodes.Single(n => n.Model.Id == inspector.Model.Id).Children,
            n => n.Key.StartsWith("pymodel:"));
        Assert.Equal(2, inspector.Components.Count);

        shell.ShowPythonModel(registered);
        foreach (var (target, expected) in new (string, Type)[]
                 { ("playground", typeof(InferencePlaygroundViewModel)), ("api", typeof(ApiConfigViewModel)), ("sandbox", typeof(ApiSandboxViewModel)), ("inspector", typeof(ModelInspectorViewModel)) })
        {
            await screen.OpenConvertedCommand.ExecuteAsync(target);
            Assert.IsType(expected, shell.CurrentViewModel);
            shell.ShowPythonModel(registered);
        }
    }

    [AvaloniaFact]
    public async Task ConversionRejectsAnInvalidOpsetAndSurfacesPythonErrors()
    {
        await using var services = UiTestSetup.Services();
        var service = new FakePythonModelService();
        var screen = CreateModelScreen(services, service, new FakePythonRuntime());
        screen.IsTrusted = true;
        await screen.LoadAsync();

        screen.OpsetText = "abc";
        await screen.ConvertAsync();
        Assert.Contains("opset", screen.ConversionError);
        Assert.Null(service.LastConversion);

        screen.OpsetText = "18";
        service.Failure = new PythonError(PythonErrorCode.UnsupportedModel, "skl2onnx cannot convert 'Foo' yet.");
        await screen.ConvertAsync();
        Assert.Contains("cannot convert", screen.ConversionError);
        Assert.False(screen.HasConversionResult);
    }

    [AvaloniaFact]
    public async Task RuntimeStatusExplainsWhatIsMissing()
    {
        await using var services = UiTestSetup.Services();
        var runtime = new FakePythonRuntime();
        var screen = CreateModelScreen(services, new FakePythonModelService(), runtime);

        await screen.RefreshRuntimeAsync();
        Assert.False(screen.RuntimeReady);
        Assert.Contains("No usable Python", screen.RuntimeStatus);

        runtime.Active = FakePythonRuntime.Interpreter(PythonSource.System, "numpy", "scikit-learn", "joblib");
        await screen.RefreshRuntimeAsync();
        Assert.True(screen.RuntimeReady);
        Assert.Contains("ONNX conversion needs: skl2onnx, onnx", screen.RuntimeStatus);

        runtime.Active = FakePythonRuntime.Interpreter(PythonSource.Managed);
        await screen.RefreshRuntimeAsync();
        Assert.False(screen.RuntimeReady);
        Assert.Contains("Install packages", screen.RuntimeStatus);
    }

    [AvaloniaFact]
    public async Task SettingsListInterpretersSelectionIsPersistedAndCustomPathsAreChecked()
    {
        await using var services = UiTestSetup.Services();
        var runtime = new FakePythonRuntime();
        var system = FakePythonRuntime.Interpreter(PythonSource.System, PythonRequirements.Managed.ToArray());
        runtime.Found.Add(system);
        var vm = new PythonRuntimeViewModel(runtime, services.GetRequiredService<IFilePickerService>(), services.GetRequiredService<IToastService>());

        await vm.RefreshAsync();

        Assert.Equal(2, vm.Interpreters.Count);
        Assert.Null(vm.SelectedInterpreter!.Interpreter);
        Assert.Empty(runtime.Selected);

        vm.SelectedInterpreter = vm.Interpreters[1];
        Assert.Equal(new PythonSelection(system.ExecutablePath, PythonSource.System), Assert.Single(runtime.Selected));
        Assert.Contains("Ready", vm.SelectedDetails);

        vm.CustomPath = "";
        await vm.UseCustomAsync();
        Assert.Contains("Enter the path", vm.Error);

        vm.CustomPath = @"C:\nowhere";
        await vm.UseCustomAsync();
        Assert.Contains("No Python interpreter", vm.Error);
        Assert.Single(runtime.Selected);

        runtime.Probe = Result<PythonInterpreter, PythonError>.Success(FakePythonRuntime.Interpreter(PythonSource.Custom, PythonRequirements.Managed.ToArray()));
        await vm.UseCustomAsync();
        Assert.Null(vm.Error);
        Assert.Equal(2, runtime.Selected.Count);
        Assert.Equal(PythonSource.Custom, runtime.Selected[1].Source);
    }

    [AvaloniaFact]
    public async Task InstallingThePythonRuntimeReportsProgressAndRefreshesTheState()
    {
        await using var services = UiTestSetup.Services();
        var runtime = new FakePythonRuntime();
        var vm = new PythonRuntimeViewModel(runtime, services.GetRequiredService<IFilePickerService>(), services.GetRequiredService<IToastService>());
        await vm.RefreshAsync();
        Assert.False(vm.CanInstallPackages);
        Assert.Equal("Download and install Python", vm.InstallPythonLabel);

        await vm.InstallPythonCommand.ExecuteAsync(null);
        Dispatcher.UIThread.RunJobs();

        Assert.True(vm.IsManagedInstalled);
        Assert.True(vm.CanInstallPackages);
        Assert.Equal("Reinstall Python", vm.InstallPythonLabel);
        Assert.Contains("Downloading Python", vm.Log);
        Assert.False(vm.IsBusy);

        await vm.InstallPackagesCommand.ExecuteAsync(null);
        Dispatcher.UIThread.RunJobs();

        Assert.Contains("Ready", vm.Status);
        Assert.Contains("scikit-learn", vm.Log);

        await vm.RemoveManagedCommand.ExecuteAsync(null);
        Assert.False(vm.IsManagedInstalled);
        Assert.Contains("No usable Python", vm.Status);
    }

    [AvaloniaFact]
    public async Task PythonScreensLoadTheirXamlAndBindings()
    {
        await using var services = UiTestSetup.Services(configure: s =>
        {
            s.AddSingleton<IPythonRuntimeService>(new FakePythonRuntime { Active = FakePythonRuntime.Interpreter(PythonSource.Managed, PythonRequirements.Managed.ToArray()) });
            s.AddSingleton<IPythonModelService>(new FakePythonModelService());
        });
        var shell = services.GetRequiredService<MainWindowViewModel>();
        var window = new Window { Width = 1280, Height = 800 };
        try
        {
            shell.OpenPythonModel(TempModel());
            var screen = Assert.IsType<PythonModelViewModel>(shell.CurrentViewModel);
            Show(new PythonModelView());
            screen.IsTrusted = true;
            await screen.LoadAsync();
            Show(new ModelInspectorView());
            screen.OpenModelCommand.Execute("playground");
            await ((InferencePlaygroundViewModel)shell.CurrentViewModel!).RunInferenceCommand.ExecuteAsync(null);
            Show(new InferencePlaygroundView());
            await screen.ConvertAsync();
            Show(new ModelInspectorView());

            shell.ShowSettings();
            Show(new SettingsView());
        }
        finally { window.Close(); }

        void Show(Control view)
        {
            view.DataContext = shell.CurrentViewModel;
            window.Content = view;
            window.Show();
            window.UpdateLayout();
            Dispatcher.UIThread.RunJobs();
            Assert.True(view.Bounds.Width > 0);
        }
    }
}
