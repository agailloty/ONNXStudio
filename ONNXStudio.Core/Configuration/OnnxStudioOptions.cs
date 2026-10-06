namespace ONNXStudio.Core.Configuration;

/// <summary>
/// Application-wide settings (from configuration or defaults).
/// </summary>
public sealed class OnnxStudioOptions
{
    public const string SectionName = "ONNXStudio";

    public int MaxModelSizeMB { get; set; } = 2048;
    public int MaxConcurrentInferences { get; set; } = 4;
    public int SessionCacheSize { get; set; } = 5;
    public string SupportedOpsetMax { get; set; } = "18";

    public OnnxStudioApiOptions Api { get; set; } = new();
}

public sealed class OnnxStudioApiOptions
{
    public int Port { get; set; } = 5000;
    public int MaxRequestSizeMB { get; set; } = 10;
    public bool EnableSwagger { get; set; } = true;
    public string CorsPolicy { get; set; } = "AllowAllForDevelopment";
}
