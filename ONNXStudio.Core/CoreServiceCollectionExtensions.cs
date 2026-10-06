using Microsoft.Extensions.DependencyInjection;
using ONNXStudio.Core.Configuration;
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
        return services;
    }
}
