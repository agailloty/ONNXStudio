using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using ONNXStudio.Api.Payloads;
using ONNXStudio.Core.Models;
using ONNXStudio.Core.Services;

namespace ONNXStudio.Api.Endpoints;

/// <summary>
/// REST endpoints exposed for the loaded models (US-005):
///   GET  /health
///   GET  /models
///   GET  /models/{id}
///   GET  /models/{id}/schema
///   POST /models/{id}/predict
/// </summary>
public static class OnnxStudioEndpoints
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    public static IEndpointRouteBuilder MapOnnxStudioEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapGet("/", (IModelRegistry registry) => Results.Ok(new
        {
            name = "ONNX Studio", version = "1.0.0", loadedModels = registry.Models.Select(m => m.Name).ToArray()
        }));
        app.MapGet("/openapi.json", (IModelRegistry registry) => Results.Text(OpenApiDocument.Create(registry.Models), "application/json"));

        app.MapGet("/health", () => Results.Ok(new { status = "ok" }));

        app.MapGet("/models", (IModelRegistry registry) =>
            Results.Ok(registry.Models.Select(ModelSummary.From).ToList()));

        app.MapGet("/models/{id}", (string id, IModelRegistry registry) =>
        {
            var model = registry.GetById(id);
            return model == null
                ? Results.NotFound(new { error = $"No model with id '{id}'" })
                : Results.Ok(ModelDetails.From(model));
        });

        app.MapGet("/models/{id}/schema", (string id, IModelRegistry registry) =>
        {
            var model = registry.GetById(id);
            return model == null
                ? Results.NotFound(new { error = $"No model with id '{id}'" })
                : Results.Ok(SchemaResponse.From(model));
        });

        app.MapPost("/models/{id}/predict", async (string id, HttpRequest request, IModelRegistry registry, IInferenceService inference) =>
        {
            var model = registry.GetById(id);
            if (model == null)
            {
                return Results.NotFound(new { error = $"No model with id '{id}'" });
            }

            JsonDocument document;
            try
            {
                document = await JsonDocument.ParseAsync(request.Body);
            }
            catch (JsonException)
            {
                return Results.BadRequest(new { error = "The request body is not valid JSON." });
            }

            using (document)
            {
                var parsed = PayloadParser.Parse(document, model);
                if (parsed.IsFailure)
                {
                    return Results.BadRequest(new { error = parsed.Error });
                }

                var result = await inference.RunAsync(model, parsed.Value!, request.HttpContext.RequestAborted);
                if (result.IsFailure)
                {
                    return Results.BadRequest(new
                    {
                        error = result.Error!.Message,
                        code = result.Error.Code.ToString()
                    });
                }

                return Results.Ok(PredictResponse.From(model, result.Value!));
            }
        });

        return app;
    }

    // ----- response DTOs -----

    public sealed class ModelSummary
    {
        public string Id { get; init; } = string.Empty;
        public string Name { get; init; } = string.Empty;
        public string Format { get; init; } = string.Empty;
        public string Producer { get; init; } = string.Empty;
        public long OpsetVersion { get; init; }
        public string FileSize { get; init; } = string.Empty;
        public int InputCount { get; init; }
        public int OutputCount { get; init; }

        public static ModelSummary From(IModel m) => new()
        {
            Id = m.Id,
            Name = m.Name,
            Format = m.Format,
            Producer = m.Producer,
            OpsetVersion = (m as OnnxModel)?.OpsetVersion ?? 0,
            FileSize = m.FileSizeDisplay,
            InputCount = m.Inputs.Count,
            OutputCount = m.Outputs.Count
        };
    }

    public sealed class ModelDetails
    {
        public string Id { get; init; } = string.Empty;
        public string Name { get; init; } = string.Empty;
        public string FilePath { get; init; } = string.Empty;
        public string Format { get; init; } = string.Empty;
        public string Producer { get; init; } = string.Empty;
        public long OpsetVersion { get; init; }
        public long IrVersion { get; init; }
        public string DocString { get; init; } = string.Empty;
        public int NodeCount { get; init; }
        public long ParameterCount { get; init; }
        public IReadOnlyList<TensorInfoDto> Inputs { get; init; } = Array.Empty<TensorInfoDto>();
        public IReadOnlyList<TensorInfoDto> Outputs { get; init; } = Array.Empty<TensorInfoDto>();

        public static ModelDetails From(IModel m) => new()
        {
            Id = m.Id,
            Name = m.Name,
            FilePath = m.FilePath,
            Format = m.Format,
            Producer = m.Producer,
            OpsetVersion = (m as OnnxModel)?.OpsetVersion ?? 0,
            IrVersion = (m as OnnxModel)?.IrVersion ?? 0,
            DocString = m.Description,
            NodeCount = m.Graph.Nodes.Count,
            ParameterCount = m.Initializers.Sum(i => i.ElementCount),
            Inputs = m.Inputs.Select(TensorInfoDto.From).ToList(),
            Outputs = m.Outputs.Select(TensorInfoDto.From).ToList()
        };
    }

    public sealed class TensorInfoDto
    {
        public string Name { get; init; } = string.Empty;
        public string Type { get; init; } = string.Empty;
        public IReadOnlyList<object> Shape { get; init; } = Array.Empty<object>();

        public static TensorInfoDto From(TensorSchema s) => new()
        {
            Name = s.Name,
            Type = s.Type.ToDisplayName(),
            Shape = s.Shape.Select(d => d.HasValue ? d.Value : (object)"?").ToList()
        };
    }

    public sealed class SchemaResponse
    {
        public IReadOnlyList<TensorInfoDto> Inputs { get; init; } = Array.Empty<TensorInfoDto>();
        public IReadOnlyList<TensorInfoDto> Outputs { get; init; } = Array.Empty<TensorInfoDto>();

        public string RequestExample { get; init; } = string.Empty;

        public static SchemaResponse From(IModel m) => new()
        {
            Inputs = m.Inputs.Select(TensorInfoDto.From).ToList(),
            Outputs = m.Outputs.Select(TensorInfoDto.From).ToList(),
            RequestExample = ExampleOrEmpty(m)
        };

        private static string ExampleOrEmpty(IModel model)
        {
            try { return ApiExamples.Payload(model); }
            catch (InvalidOperationException) { return string.Empty; }
        }
    }

    public sealed class PredictResponse
    {
        public string ModelId { get; init; } = string.Empty;
        public long ExecutionTimeMs { get; init; }
        public Dictionary<string, object> Outputs { get; init; } = new();
        public Dictionary<string, IReadOnlyList<long>> OutputShapes { get; init; } = new();

        public static PredictResponse From(IModel model, InferenceResult result)
        {
            var outputs = new Dictionary<string, object>();
            var shapes = new Dictionary<string, IReadOnlyList<long>>();

            foreach (var output in result.Outputs)
            {
                outputs[output.Name] = output.Data.Length == 1
                    ? ConvertToScalar(output)
                    : output.Data;
                shapes[output.Name] = output.Shape.ToList();

                // For classification-style rank-2 outputs [1, N], expose a flat copy too
                if (output.Shape.Count == 2 && output.Shape[0] == 1)
                {
                    outputs[output.Name + "_flat"] = output.Data;
                }
            }

            return new PredictResponse
            {
                ModelId = model.Id,
                ExecutionTimeMs = result.ExecutionTimeMs,
                Outputs = outputs,
                OutputShapes = shapes
            };
        }

        private static object ConvertToScalar(TensorOutput output)
        {
            var value = output.Data.GetValue(0)!;
            return value;
        }
    }

    private static string ToJson(this object obj) => JsonSerializer.Serialize(obj, JsonOptions);
}
