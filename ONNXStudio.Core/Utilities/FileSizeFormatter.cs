namespace ONNXStudio.Core.Utilities;

/// <summary>
/// Formats byte counts into human-readable sizes.
/// </summary>
public static class FileSizeFormatter
{
    private static readonly string[] Units = { "B", "KB", "MB", "GB", "TB" };

    public static string Format(long bytes)
    {
        double len = bytes;
        var order = 0;
        while (len >= 1024 && order < Units.Length - 1)
        {
            order++;
            len /= 1024;
        }
        return $"{len:0.#} {Units[order]}";
    }
}
