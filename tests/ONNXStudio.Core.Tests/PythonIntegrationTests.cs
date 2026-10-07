using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Python;
using ONNXStudio.Core.Services;
using Xunit;

namespace ONNXStudio.Core.Tests;

/// <summary>
/// Runs only when ONNXSTUDIO_PYTHON_INTEGRATION=1. Downloads the managed Python and
/// installs scikit-learn / skl2onnx (several hundred MB) into
/// ONNXSTUDIO_TEST_PYTHON_DIR (default: a temp folder, deleted afterwards).
/// </summary>
public sealed class IntegrationFactAttribute : FactAttribute
{
    public const string Variable = "ONNXSTUDIO_PYTHON_INTEGRATION";

    public IntegrationFactAttribute()
    {
        if (Environment.GetEnvironmentVariable(Variable) != "1")
            Skip = $"Set {Variable}=1 to run the Python integration tests (downloads Python and packages).";
    }
}

public sealed class PythonIntegrationFixture : IAsyncLifetime
{
    private bool _ownsDirectory;

    public string Directory { get; private set; } = string.Empty;
    public string ModelsDirectory => Path.Combine(Directory, "models");
    public IPythonRuntimeService Runtime { get; private set; } = null!;
    public IPythonModelService Models { get; private set; } = null!;
    public PythonInterpreter Interpreter { get; private set; } = null!;

    public async Task InitializeAsync()
    {
        if (Environment.GetEnvironmentVariable(IntegrationFactAttribute.Variable) != "1") return;

        var configured = Environment.GetEnvironmentVariable("ONNXSTUDIO_TEST_PYTHON_DIR");
        _ownsDirectory = string.IsNullOrEmpty(configured);
        Directory = configured ?? Path.Combine(Path.GetTempPath(), "onnxstudio-py-" + Guid.NewGuid().ToString("N"));

        var options = new OnnxStudioOptions { Python = new OnnxStudioPythonOptions { Directory = Path.Combine(Directory, "python") } };
        var services = PythonTestServices.Create(options);
        Runtime = services.Runtime;
        Models = services.Models;

        var active = await Runtime.GetActiveAsync();
        if (active is not { Source: PythonSource.Managed } || active.MissingManaged.Count > 0)
        {
            var progress = new Progress<PythonInstallProgress>();
            var python = await Runtime.InstallPythonAsync(progress);
            Assert.True(python.IsSuccess, python.Error?.ToString() + python.Error?.TechnicalDetails);
            var packages = await Runtime.InstallPackagesAsync(progress);
            Assert.True(packages.IsSuccess, packages.Error?.ToString() + packages.Error?.TechnicalDetails);
            active = packages.Value;
        }
        Interpreter = active!;

        System.IO.Directory.CreateDirectory(ModelsDirectory);
        var script = Path.Combine(ModelsDirectory, "make_models.py");
        await File.WriteAllTextAsync(script, MakeModelsScript);
        var run = await new ProcessRunner().RunAsync(new ProcessSpec(
            Interpreter.ExecutablePath, new[] { script, ModelsDirectory }, Environment: PythonEnvironment.For(PythonSource.Managed)));
        Assert.True(run.Succeeded, run.StdErr);
    }

    public Task DisposeAsync()
    {
        if (_ownsDirectory && Directory.Length > 0)
        {
            try { System.IO.Directory.Delete(Directory, recursive: true); } catch (IOException) { }
        }
        return Task.CompletedTask;
    }

    private const string MakeModelsScript = """
        import sys, os, pickle, joblib
        import pandas as pd
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_extraction.text import TfidfVectorizer
        out = sys.argv[1]
        X, y = load_iris(return_X_y=True)
        joblib.dump(RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y), os.path.join(out, "rf.joblib"))
        with open(os.path.join(out, "pipe.pkl"), "wb") as f:
            pickle.dump(Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]).fit(X, y), f)
        df = pd.DataFrame(X, columns=["sl", "sw", "pl", "pw"])
        joblib.dump(LinearRegression().fit(df, y), os.path.join(out, "named.joblib"))
        texts = ["good great fine", "bad awful poor", "great excellent", "terrible bad", "fine good"]
        labels = [1, 0, 1, 0, 1]
        joblib.dump(Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())]).fit(texts, labels), os.path.join(out, "text.joblib"))
        with open(os.path.join(out, "missing.pkl"), "wb") as f:
            f.write(b"cnonexistent_module_xyz\nFoo\n.")
        with open(os.path.join(out, "garbage.pkl"), "wb") as f:
            f.write(b"this is not a pickle")
        """;
}

public static class PythonTestServices
{
    public static (IPythonRuntimeService Runtime, IPythonModelService Models) Create(OnnxStudioOptions options)
    {
        var paths = PythonPaths.From(options);
        var runner = new ProcessRunner();
        var downloader = new PythonDownloader();
        var probe = new PythonProbe(runner);
        var installer = new PythonRuntimeInstaller(paths, new PythonDistributionResolver(downloader, options.Python), downloader, runner,
            NullLogger<PythonRuntimeInstaller>.Instance);
        var runtime = new PythonRuntimeService(paths, new PythonDiscovery(runner), probe, installer, NullLogger<PythonRuntimeService>.Instance);
        var worker = new PythonWorkerClient(paths, runner, options.Python, NullLogger<PythonWorkerClient>.Instance);
        return (runtime, new PythonModelService(runtime, worker));
    }
}

public class PythonIntegrationTests : IClassFixture<PythonIntegrationFixture>
{
    private readonly PythonIntegrationFixture _fx;

    public PythonIntegrationTests(PythonIntegrationFixture fixture) => _fx = fixture;

    private string Model(string name) => Path.Combine(_fx.ModelsDirectory, name);

    [IntegrationFact]
    public async Task ManagedRuntimeHasAllPackagesAndIsSelectedAutomatically()
    {
        Assert.Equal(PythonSource.Managed, _fx.Interpreter.Source);
        Assert.True(_fx.Interpreter.CanConvert);
        Assert.Empty(_fx.Interpreter.MissingManaged);
        Assert.Equal(_fx.Interpreter.ExecutablePath, (await _fx.Runtime.GetActiveAsync())!.ExecutablePath);
    }

    [IntegrationFact]
    public async Task InspectDescribesAClassifier()
    {
        var result = await _fx.Models.InspectAsync(Model("rf.joblib"), trustConfirmed: true);
        Assert.True(result.IsSuccess, result.Error?.TechnicalDetails);
        var info = result.Value!;
        Assert.Equal("RandomForestClassifier", info.ClassName);
        Assert.Equal(4, info.FeatureCount);
        Assert.Equal(new[] { "0", "1", "2" }, info.Classes);
        Assert.Contains("predict_proba", info.Methods);
        Assert.True(info.IsClassifier);
    }

    [IntegrationFact]
    public async Task InspectDescribesAPickledPipeline()
    {
        var info = (await _fx.Models.InspectAsync(Model("pipe.pkl"), true)).Value!;
        Assert.True(info.IsPipeline);
        Assert.Equal(new[] { "StandardScaler", "LogisticRegression" }, info.Steps.Select(s => s.ClassName));
        Assert.Equal(4, info.FeatureCount);
    }

    [IntegrationFact]
    public async Task PredictReturnsLabelsAndProbabilities()
    {
        var result = await _fx.Models.PredictAsync(new PythonPredictionRequest(
            Model("rf.joblib"), true, "all", null,
            new[] { (IReadOnlyList<string>)new[] { "5.1", "3.5", "1.4", "0.2" }, new[] { "6.7", "3.0", "5.2", "2.3" } }));
        Assert.True(result.IsSuccess, result.Error?.TechnicalDetails);
        var predict = result.Value!.Outputs.Single(o => o.Name == "predict");
        Assert.Equal(new[] { "0", "2" }, predict.Rows);
        var proba = result.Value.Outputs.Single(o => o.Name == "predict_proba");
        Assert.Equal(2, proba.Rows.Count);
        Assert.Equal(new[] { 2, 3 }, proba.Shape);
    }

    [IntegrationFact]
    public async Task PredictUsesColumnNamesOfDataFrameTrainedModels()
    {
        var result = await _fx.Models.PredictAsync(new PythonPredictionRequest(
            Model("named.joblib"), true, "auto", new[] { "pw", "pl", "sw", "sl" },
            new[] { (IReadOnlyList<string>)new[] { "0.2", "1.4", "3.5", "5.1" } }));
        Assert.True(result.IsSuccess, result.Error?.TechnicalDetails);
        Assert.Empty(result.Value!.Warnings);
        Assert.InRange(double.Parse(result.Value.Outputs[0].Rows[0], System.Globalization.CultureInfo.InvariantCulture), -0.5, 0.5);
    }

    [IntegrationFact]
    public async Task PredictAcceptsRawTextForTextPipelines()
    {
        var result = await _fx.Models.PredictAsync(new PythonPredictionRequest(
            Model("text.joblib"), true, "predict", null,
            new[] { (IReadOnlyList<string>)new[] { "great good" }, new[] { "awful bad" } }));
        Assert.True(result.IsSuccess, result.Error?.TechnicalDetails);
        Assert.Equal(new[] { "1", "0" }, result.Value!.Outputs[0].Rows);
    }

    [IntegrationFact]
    public async Task PredictWithWrongFeatureCountFailsWithAMessage()
    {
        var result = await _fx.Models.PredictAsync(new PythonPredictionRequest(
            Model("rf.joblib"), true, "predict", null, new[] { (IReadOnlyList<string>)new[] { "1", "2" } }));
        Assert.True(result.IsFailure);
        Assert.Equal(PythonErrorCode.PredictionFailed, result.Error!.Code);
        Assert.Contains("feature", result.Error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [IntegrationFact]
    public async Task MissingModuleAndGarbageFilesProduceFriendlyErrors()
    {
        var missing = await _fx.Models.InspectAsync(Model("missing.pkl"), true);
        Assert.Equal(PythonErrorCode.MissingModule, missing.Error!.Code);
        Assert.Contains("nonexistent_module_xyz", missing.Error.Message);

        var garbage = await _fx.Models.InspectAsync(Model("garbage.pkl"), true);
        Assert.Equal(PythonErrorCode.LoadFailed, garbage.Error!.Code);
    }

    [IntegrationFact]
    public async Task ConvertedClassifierLoadsInONNXStudioAndMatchesScikitLearn()
    {
        var output = Path.Combine(_fx.ModelsDirectory, "rf.onnx");
        var converted = await _fx.Models.ConvertAsync(new PythonConversionRequest(Model("rf.joblib"), output, true));
        Assert.True(converted.IsSuccess, converted.Error?.TechnicalDetails);
        Assert.True(converted.Value!.TargetOpset <= 18);
        Assert.True(converted.Value.Validation.Performed);
        Assert.True(converted.Value.Validation.Passed, converted.Value.Validation.Detail);

        var loader = TestSetup.CreateLoader();
        var loaded = await loader.LoadAsync(output);
        Assert.True(loaded.IsSuccess, loaded.Error?.ToString());
        var onnx = loaded.Value!;
        Assert.Single(onnx.Inputs);

        var sessions = new InferenceSessionManager(Options.Create(new OnnxStudioOptions()));
        var inference = new InferenceService(sessions, Options.Create(new OnnxStudioOptions()), NullLogger<InferenceService>.Instance);
        var run = await inference.RunAsync(onnx, new Dictionary<string, InferenceInputValue>
        {
            [onnx.Inputs[0].Name] = InferenceInputValue.Tensor(new[] { 6.7f, 3.0f, 5.2f, 2.3f }, new long[] { 1, 4 })
        });
        Assert.True(run.IsSuccess, run.Error?.TechnicalDetails);
        var label = run.Value!.Outputs.First(o => o.Name == "label");
        Assert.Equal(2L, Convert.ToInt64(label.Data.GetValue(0)));
    }

    [IntegrationFact]
    public async Task ConvertSupportsPipelinesNamedColumnsAndText()
    {
        foreach (var name in new[] { "pipe.pkl", "named.joblib", "text.joblib" })
        {
            var output = Path.Combine(_fx.ModelsDirectory, Path.GetFileNameWithoutExtension(name) + ".onnx");
            var converted = await _fx.Models.ConvertAsync(new PythonConversionRequest(Model(name), output, true));
            Assert.True(converted.IsSuccess, name + ": " + converted.Error?.Message + converted.Error?.TechnicalDetails);
            var loaded = await TestSetup.CreateLoader().LoadAsync(output);
            Assert.True(loaded.IsSuccess, name + ": " + loaded.Error?.ToString());
            if (!name.StartsWith("text")) Assert.True(converted.Value!.Validation.Passed, name + ": " + converted.Value.Validation.Detail);
        }
    }

    [IntegrationFact]
    public async Task ConvertHonoursRequestedOpsetAndCapsToSupportedMaximum()
    {
        var output = Path.Combine(_fx.ModelsDirectory, "rf-opset13.onnx");
        var converted = await _fx.Models.ConvertAsync(new PythonConversionRequest(Model("rf.joblib"), output, true, TargetOpset: 13));
        Assert.True(converted.IsSuccess, converted.Error?.TechnicalDetails);
        Assert.Equal(13, converted.Value!.TargetOpset);
    }
}
