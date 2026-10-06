using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using ONNXStudio.Api;

namespace ONNXStudio.Api;

/// <summary>
/// DI registrations for the Api module.
/// </summary>
public static class ApiServiceCollectionExtensions
{
    public static IServiceCollection AddOnnxStudioApi(this IServiceCollection services)
    {
        services.AddSingleton<ApiServerHost>(sp => new ApiServerHost(
            sp.GetRequiredService<Core.Services.IModelRegistry>(),
            sp.GetRequiredService<Core.Services.IInferenceService>(),
            sp.GetRequiredService<Microsoft.Extensions.Options.IOptions<Core.Configuration.OnnxStudioOptions>>(),
            sp.GetService<ILogger<ApiServerHost>>()));
        return services;
    }
}
