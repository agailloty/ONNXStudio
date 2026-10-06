using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using ONNXStudio.Core.Configuration;
using ONNXStudio.Core.Services;

namespace ONNXStudio.Core.Tests;

/// <summary>
/// Shared helpers for Core tests (fixture paths, loader factory).
/// </summary>
public static class TestSetup
{
    public static string FixturePath(string name)
        => Path.Combine(AppContext.BaseDirectory, "fixtures", name);

    public static ModelLoader CreateLoader(OnnxStudioOptions? options = null)
    {
        options ??= new OnnxStudioOptions();
        return new ModelLoader(Options.Create(options), NullLogger<ModelLoader>.Instance);
    }
}
