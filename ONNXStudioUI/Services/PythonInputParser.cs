using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ONNXStudio.Core.Models;

namespace ONNXStudioUI.Services;

/// <summary>Parsed tabular input of the Python model screen.</summary>
public sealed record PythonTableInput(IReadOnlyList<string>? Columns, IReadOnlyList<IReadOnlyList<string>> Rows);

/// <summary>
/// Turns the text typed in the Python inference form (CSV-like: one row per line)
/// into rows of string cells; the Python worker converts cells to numbers.
/// </summary>
public static class PythonInputParser
{
    public static Result<PythonTableInput, string> Parse(string? text, bool hasHeader, bool singleTextColumn)
    {
        var lines = (text ?? string.Empty)
            .Split('\n')
            .Select(l => l.TrimEnd('\r'))
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .ToList();
        if (lines.Count == 0)
            return Fail("Enter at least one row of values.");

        IReadOnlyList<string>? columns = null;
        if (hasHeader)
        {
            if (singleTextColumn) return Fail("Text models take one text per line: turn off 'First line is a header'.");
            columns = SplitLine(lines[0], DetectDelimiter(lines[0])).Select(c => c.Trim()).ToArray();
            lines.RemoveAt(0);
            if (lines.Count == 0) return Fail("The input only contains a header line.");
        }

        var rows = new List<IReadOnlyList<string>>();
        if (singleTextColumn)
        {
            rows.AddRange(lines.Select(l => (IReadOnlyList<string>)new[] { l.Trim() }));
        }
        else
        {
            var delimiter = DetectDelimiter(lines[0]);
            foreach (var line in lines) rows.Add(SplitLine(line, delimiter).Select(c => c.Trim()).ToArray());

            var width = rows[0].Count;
            var bad = rows.FindIndex(r => r.Count != width);
            if (bad >= 0)
                return Fail($"Row {bad + 1 + (hasHeader ? 1 : 0)} has {rows[bad].Count} values but the first row has {width}.");
            if (columns != null && columns.Count != width)
                return Fail($"The header has {columns.Count} columns but the rows have {width} values.");
        }

        return Result<PythonTableInput, string>.Success(new PythonTableInput(columns, rows));
    }

    private static char DetectDelimiter(string line)
        => line.Contains('\t') ? '\t' : line.Contains(';') ? ';' : line.Contains(',') ? ',' : ' ';

    /// <summary>Splits on the delimiter, honouring double quoted cells ("a, b" and "" escapes).</summary>
    internal static IEnumerable<string> SplitLine(string line, char delimiter)
    {
        var cell = new StringBuilder();
        var quoted = false;
        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];
            if (quoted)
            {
                if (c == '"' && i + 1 < line.Length && line[i + 1] == '"') { cell.Append('"'); i++; }
                else if (c == '"') quoted = false;
                else cell.Append(c);
            }
            else if (c == '"') quoted = true;
            else if (c == delimiter)
            {
                if (delimiter == ' ' && cell.Length == 0) continue;
                yield return cell.ToString();
                cell.Clear();
            }
            else cell.Append(c);
        }
        if (delimiter != ' ' || cell.Length > 0) yield return cell.ToString();
    }

    private static Result<PythonTableInput, string> Fail(string message)
        => Result<PythonTableInput, string>.Failure(message);
}
