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

    /// <summary>ONNX models and Python (joblib / pickle) models.</summary>
    Task<string[]> PickModelFilesAsync();
    Task<string?> PickImageFileAsync();

    /// <summary>A Python interpreter executable.</summary>
    Task<string?> PickPythonInterpreterAsync();

    /// <summary>A folder (e.g. a virtual environment or a Python installation).</summary>
    Task<string?> PickFolderAsync(string title);

    /// <summary>Where to write a converted .onnx file; null when cancelled.</summary>
    Task<string?> PickOnnxSavePathAsync(string suggestedFileName, string? initialDirectory);
}

public sealed class FilePickerService : IFilePickerService
{
    private Func<IStorageProvider>? _storageProvider;

    private static readonly string[] PythonPatterns = ONNXStudio.Core.Python.PythonModel.SupportedExtensions
        .Select(e => "*" + e).ToArray();

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
            Title = "Open model",
            AllowMultiple = true,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("All models") { Patterns = new[] { "*.onnx" }.Concat(PythonPatterns).ToArray() },
                new FilePickerFileType("ONNX model") { Patterns = new[] { "*.onnx" } },
                new FilePickerFileType("Python model (joblib / pickle)") { Patterns = PythonPatterns }
            }
        });

        return result.Count == 0
            ? Array.Empty<string>()
            : result
                .Select(f => f.TryGetLocalPath() ?? f.Path.AbsolutePath)
                .ToArray();
    }

    public async Task<string?> PickPythonInterpreterAsync()
    {
        var provider = _storageProvider?.Invoke();
        if (provider == null) return null;

        var result = await provider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select a Python interpreter",
            AllowMultiple = false
        });
        return result.Count == 0 ? null : result[0].TryGetLocalPath() ?? result[0].Path.AbsolutePath;
    }

    public async Task<string?> PickFolderAsync(string title)
    {
        var provider = _storageProvider?.Invoke();
        if (provider == null) return null;

        var result = await provider.OpenFolderPickerAsync(new FolderPickerOpenOptions { Title = title, AllowMultiple = false });
        return result.Count == 0 ? null : result[0].TryGetLocalPath() ?? result[0].Path.AbsolutePath;
    }

    public async Task<string?> PickOnnxSavePathAsync(string suggestedFileName, string? initialDirectory)
    {
        var provider = _storageProvider?.Invoke();
        if (provider == null) return null;

        var options = new FilePickerSaveOptions
        {
            Title = "Save ONNX model",
            SuggestedFileName = suggestedFileName,
            DefaultExtension = "onnx",
            ShowOverwritePrompt = true,
            FileTypeChoices = new[] { new FilePickerFileType("ONNX model") { Patterns = new[] { "*.onnx" } } }
        };
        if (initialDirectory != null && Directory.Exists(initialDirectory))
        {
            options.SuggestedStartLocation = await provider.TryGetFolderFromPathAsync(initialDirectory);
        }

        var file = await provider.SaveFilePickerAsync(options);
        return file == null ? null : file.TryGetLocalPath() ?? file.Path.AbsolutePath;
    }

    public async Task<string?> PickImageFileAsync()
    {
        var provider = _storageProvider?.Invoke();
        if (provider == null)
        {
            return null;
        }

        var result = await provider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Open image",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("Image") { Patterns = new[] { "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp" } }
            }
        });

        return result.Count == 0
            ? null
            : result[0].TryGetLocalPath() ?? result[0].Path.AbsolutePath;
    }
}
