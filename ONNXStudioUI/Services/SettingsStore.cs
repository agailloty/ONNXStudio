using System.Text.Json.Nodes;

namespace ONNXStudioUI.Services;

public sealed class SettingsStore
{
    private readonly string _path;
    public SettingsStore() : this(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ONNXStudio", "settings.json")) { }
    public SettingsStore(string path) => _path = path;

    public (AppTheme Theme, int Port) Load()
    {
        try
        {
            var json = JsonNode.Parse(File.ReadAllText(_path));
            if (Enum.TryParse<AppTheme>(json?["theme"]?.GetValue<string>(), out var theme) && Enum.IsDefined(theme)
                && json?["apiPort"]?.GetValue<int>() is int port && port is >= 0 and <= 65535)
                return (theme, port);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.Text.Json.JsonException or InvalidOperationException or FormatException) { }
        return (AppTheme.Dark, 5000);
    }

    public void Save(AppTheme theme, int port)
    {
        if (port is < 0 or > 65535) throw new ArgumentOutOfRangeException(nameof(port));
        Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
        var temp = _path + ".tmp";
        File.WriteAllText(temp, new JsonObject { ["theme"] = theme.ToString(), ["apiPort"] = port }.ToJsonString());
        File.Move(temp, _path, overwrite: true);
    }
}
