namespace ONNXStudio.Core.Models;

/// <summary>
/// Functional result type: Success(value) or Failure(error).
/// Used across Core services to avoid exception-driven control flow.
/// </summary>
public readonly struct Result<TValue, TError>
    where TError : notnull
{
    public bool IsSuccess { get; }
    public bool IsFailure => !IsSuccess;

    public TValue? Value { get; }
    public TError? Error { get; }

    private Result(TValue value)
    {
        IsSuccess = true;
        Value = value;
        Error = default;
    }

    private Result(TError error)
    {
        IsSuccess = false;
        Value = default;
        Error = error;
    }

    public static Result<TValue, TError> Success(TValue value) => new(value);
    public static Result<TValue, TError> Failure(TError error) => new(error);

    public TOut Match<TOut>(Func<TValue, TOut> onSuccess, Func<TError, TOut> onFailure)
        => IsSuccess ? onSuccess(Value!) : onFailure(Error!);

    public static implicit operator Result<TValue, TError>(TValue value) => Success(value);
}
