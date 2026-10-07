using System.Collections;
using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace ONNXStudioUI.ViewModels.Screens;

/// <summary>
/// Formats only the visible page. Tree ensembles can contain millions of values
/// in a single attribute; neither formatting nor text layout may visit them all.
/// </summary>
public partial class AttributeEntry : ObservableObject
{
    public const int PageSize = 16;
    public const int MaxTextLength = 256;
    private readonly object _value;
    private int _page;

    public AttributeEntry(string key, object value) { Key = key; _value = value; }

    public string Key { get; }
    public int Count => _value is IList list ? list.Count : 1;
    public bool HasMultiplePages => Count > PageSize && _value is IList;
    public string PageSummary => HasMultiplePages
        ? $"{_page * PageSize + 1:N0}–{Math.Min((_page + 1) * PageSize, Count):N0} of {Count:N0} values"
        : string.Empty;

    public string Value
    {
        get
        {
            if (_value is not IList list) return FormatScalar(_value);
            int start = _page * PageSize, end = Math.Min(start + PageSize, list.Count);
            var values = new string[end - start];
            for (int i = start; i < end; i++) values[i - start] = FormatScalar(list[i]);
            return "[" + string.Join(", ", values) + "]";
        }
    }

    private static string FormatScalar(object? value)
    {
        var text = value is IFormattable number
            ? number.ToString(null, CultureInfo.InvariantCulture)
            : value?.ToString() ?? string.Empty;
        return text.Length > MaxTextLength ? text[..MaxTextLength] + "… (truncated)" : text;
    }

    private bool CanPreviousPage() => _page > 0;
    private bool CanNextPage() => (_page + 1) * PageSize < Count && _value is IList;

    [RelayCommand(CanExecute = nameof(CanPreviousPage))]
    private void PreviousPage() => ChangePage(-1);

    [RelayCommand(CanExecute = nameof(CanNextPage))]
    private void NextPage() => ChangePage(1);

    private void ChangePage(int delta)
    {
        _page += delta;
        OnPropertyChanged(nameof(Value));
        OnPropertyChanged(nameof(PageSummary));
        PreviousPageCommand.NotifyCanExecuteChanged();
        NextPageCommand.NotifyCanExecuteChanged();
    }
}
