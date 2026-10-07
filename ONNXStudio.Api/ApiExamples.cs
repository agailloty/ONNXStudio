using System.Text.Json;
using System.Text.Json.Nodes;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Api;

/// <summary>Schema-derived examples shared by API configuration and sandbox.</summary>
public static class ApiExamples
{
    private static readonly JsonSerializerOptions Pretty = new() { WriteIndented = true };

    public static string Payload(OnnxModel model)
    {
        var inputs = new JsonObject();
        foreach (var input in model.Inputs)
        {
            var shape = input.Shape.Select(d => d ?? 1).ToArray();
            long count = 1;
            foreach (var dimension in shape)
            {
                if (dimension <= 0 || dimension > 1_000_000 / count)
                    throw new InvalidOperationException($"Input '{input.Name}' is too large for an automatic example. Enter its data manually.");
                count *= dimension;
            }
            inputs[input.Name] = Sample(input.Type, shape, 0);
        }
        return new JsonObject { ["inputs"] = inputs }.ToJsonString(Pretty);
    }

    private static JsonNode Sample(DataType type, long[] shape, int depth)
    {
        if (depth == shape.Length)
            return type == DataType.String ? JsonValue.Create("sample")!
                : type == DataType.Bool ? JsonValue.Create(false)! : JsonValue.Create(0)!;
        var array = new JsonArray();
        for (long i = 0; i < shape[depth]; i++) array.Add(Sample(type, shape, depth + 1));
        return array;
    }

    public static string Schema(OnnxModel model)
    {
        var properties = new JsonObject();
        var required = new JsonArray();
        foreach (var input in model.Inputs)
        {
            JsonObject node = new() { ["type"] = input.Type == DataType.String ? "string" : input.Type == DataType.Bool ? "boolean" : input.Type is DataType.Float32 or DataType.Float64 or DataType.Float16 ? "number" : "integer" };
            foreach (var dimension in input.Shape.Reverse())
            {
                node = new JsonObject { ["type"] = "array", ["items"] = node };
                if (dimension.HasValue) { node["minItems"] = dimension.Value; node["maxItems"] = dimension.Value; }
            }
            node["description"] = input.Display;
            properties[input.Name] = node;
            required.Add(input.Name);
        }
        return new JsonObject
        {
            ["$schema"] = "http://json-schema.org/draft-07/schema#",
            ["title"] = model.Name + " predict request",
            ["type"] = "object",
            ["required"] = new JsonArray("inputs"),
            ["properties"] = new JsonObject
            {
                ["inputs"] = new JsonObject
                { ["type"] = "object", ["required"] = required, ["properties"] = properties, ["additionalProperties"] = false }
            }
        }.ToJsonString(Pretty);
    }

    public static string Curl(string method, string url, string body) =>
        $"curl -X {method} '{url.Replace("'", "'\\''")}'" +
        (method == "POST" ? " -H 'Content-Type: application/json' --data '" + body.Replace("'", "'\\''") + "'" : "");
}
