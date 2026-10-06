namespace ONNXStudio.Core.Models;

/// <summary>
/// Machine-readable reason of a model load failure.
/// </summary>
public enum ModelLoadErrorCode
{
    None,
    FileNotFound,
    InvalidFileExtension,
    InvalidFileFormat,
    FileTooLarge,
    UnsupportedOpsetVersion,
    InvalidModel,
    LoadFailed
}

/// <summary>
/// User-friendly load error: a short actionable message, plus technical
/// details reserved for logs and an optional original exception.
/// </summary>
public sealed class ModelLoadError
{
    public ModelLoadErrorCode Code { get; }
    public string Message { get; }
    public string TechnicalDetails { get; }
    public Exception? OriginalException { get; }

    public ModelLoadError(
        ModelLoadErrorCode code,
        string message,
        string technicalDetails = "",
        Exception? originalException = null)
    {
        Code = code;
        Message = message;
        TechnicalDetails = technicalDetails;
        OriginalException = originalException;
    }

    public override string ToString() => $"[{Code}] {Message}";
}
