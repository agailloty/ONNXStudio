using System;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Platform.Storage;

namespace ONNXStudioUI.Services;

/// <summary>
/// Abstraction over the Avalonia storage picker so view models stay testable.
/// </summary>
public interface IFilePickerService
{
    Task<string?> PickModelFileAsync();
    Task<string[]> PickModelFilesAsync();
}

public sealed class FilePickerService : IFilePickerService
{
    private Func<IStorageProvider>? _storageProvider;

    /// <summary>
    /// Wired by the main window once the TopLevel is available.
    /// </summary>
    public void Initialize(Func<IStorageProvider> storageProvider)
    {
        _storageProvider = storageProvider;
    }

    public async Task<string?> PickModelFileAsync()
    {
        var files = await PickModelFilesAsync().ConfigureAwait(false);
        return files.Length > 0 ? files[0] : null;
    }

    public async Task<string[]> PickModelFilesAsync()
    {
        var provider = _storageProvider?.Invoke();
        if (provider == null)
        {
            return Array.Empty<string>();
        }

        var result = await provider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Open ONNX model",
            AllowMultiple = true,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("ONNX model") { Patterns = new[] { "*.onnx" } }
            }
        });

        return result.Count == 0
            ? Array.Empty<string>()
            : result
                .Select(f => f.TryGetLocalPath() ?? f.Path.AbsolutePath)
                .ToArray();
    }
}
