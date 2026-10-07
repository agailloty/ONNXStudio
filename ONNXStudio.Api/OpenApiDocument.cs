using System.Text.Json.Nodes;
using ONNXStudio.Core.Models;

namespace ONNXStudio.Api;

/// <summary>OpenAPI document derived from the models currently in the shared registry.</summary>
public static class OpenApiDocument
{
    public static string Create(IReadOnlyList<OnnxModel> models)
    {
        var paths = new JsonObject
        {
            ["/"] = Get("Application information"),
            ["/health"] = Get("Server health"),
            ["/models"] = Get("Loaded models")
        };
        foreach (var model in models)
        {
            var path = "/models/" + Uri.EscapeDataString(model.Id);
            paths[path] = Get("Metadata for " + model.Name);
            paths[path + "/schema"] = Get("Input and output schemas for " + model.Name);
            var schema = JsonNode.Parse(ApiExamples.Schema(model))!.AsObject();
            schema.Remove("$schema");
            paths[path + "/predict"] = new JsonObject
            {
                ["post"] = new JsonObject
                {
                    ["summary"] = "Run " + model.Name,
                    ["requestBody"] = new JsonObject
                    {
                        ["required"] = true,
                        ["content"] = new JsonObject
                        { ["application/json"] = new JsonObject { ["schema"] = schema } }
                    },
                    ["responses"] = new JsonObject
                    {
                        ["200"] = new JsonObject { ["description"] = "Output tensors and execution time" },
                        ["400"] = new JsonObject { ["description"] = "Invalid JSON, tensor type or shape" },
                        ["404"] = new JsonObject { ["description"] = "Model is no longer loaded" }
                    }
                }
            };
        }
        return new JsonObject
        {
            ["openapi"] = "3.1.0",
            ["info"] = new JsonObject { ["title"] = "ONNX Studio", ["version"] = "1.0.0" },
            ["paths"] = paths
        }.ToJsonString();
    }

    private static JsonObject Get(string summary) => new()
    {
        ["get"] = new JsonObject
        {
            ["summary"] = summary,
            ["responses"] = new JsonObject
            { ["200"] = new JsonObject { ["description"] = "Success" } }
        }
    };
}
