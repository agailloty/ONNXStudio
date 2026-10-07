using Microsoft.Extensions.DependencyInjection;
using ONNXStudio.Core.Configuration;
using Microsoft.Extensions.Options;
using ONNXStudio.Core.Python;
using ONNXStudio.Core.Services;

namespace ONNXStudio.Core;

/// <summary>
/// DI registrations for the Core module.
/// </summary>
public static class CoreServiceCollectionExtensions
{
    public static IServiceCollection AddOnnxStudioCore(this IServiceCollection services)
    {
        services.AddOptions<OnnxStudioOptions>();
        services.AddSingleton<ModelRegistry>();
        services.AddSingleton<IModelRegistry>(sp => sp.GetRequiredService<ModelRegistry>());
        services.AddSingleton<IModelLoader, ModelLoader>();
        services.AddSingleton<IInferenceSessionManager, InferenceSessionManager>();
        services.AddSingleton<IInferenceService, InferenceService>();
        services.AddSingleton<IFormGenerationService, FormGenerationService>();
        services.AddSingleton<IGraphAnalysisService, GraphAnalysisService>();
        services.AddOnnxStudioPython();
        return services;
    }

    /// <summary>
    /// Python support: runtime discovery / installation and joblib-pickle models.
    /// </summary>
    public static IServiceCollection AddOnnxStudioPython(this IServiceCollection services)
    {
        services.AddSingleton(sp => sp.GetRequiredService<IOptions<OnnxStudioOptions>>().Value.Python);
        services.AddSingleton(sp => PythonPaths.From(sp.GetRequiredService<IOptions<OnnxStudioOptions>>().Value));
        services.AddSingleton<IProcessRunner, ProcessRunner>();
        services.AddSingleton<IPythonDownloader, PythonDownloader>();
        services.AddSingleton<PythonDistributionResolver>();
        services.AddSingleton<IPythonProbe, PythonProbe>();
        services.AddSingleton<IPythonDiscovery, PythonDiscovery>();
        services.AddSingleton<IPythonRuntimeInstaller, PythonRuntimeInstaller>();
        services.AddSingleton<IPythonRuntimeService, PythonRuntimeService>();
        services.AddSingleton<IPythonWorkerClient, PythonWorkerClient>();
        services.AddSingleton<IPythonModelService, PythonModelService>();
        services.AddSingleton<IPythonModelRegistry, PythonModelRegistry>();
        return services;
    }
}