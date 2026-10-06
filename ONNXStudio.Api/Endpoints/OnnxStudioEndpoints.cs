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

                var result = await inference.RunAsync(model, parsed.Value!);
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
        public string Producer { get; init; } = string.Empty;
        public long OpsetVersion { get; init; }
        public string FileSize { get; init; } = string.Empty;
        public int InputCount { get; init; }
        public int OutputCount { get; init; }

        public static ModelSummary From(OnnxModel m) => new()
        {
            Id = m.Id,
            Name = m.Name,
            Producer = m.ProducerName,
            OpsetVersion = m.OpsetVersion,
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
        public string Producer { get; init; } = string.Empty;
        public long OpsetVersion { get; init; }
        public long IrVersion { get; init; }
        public string DocString { get; init; } = string.Empty;
        public int NodeCount { get; init; }
        public long ParameterCount { get; init; }
        public IReadOnlyList<TensorInfoDto> Inputs { get; init; } = Array.Empty<TensorInfoDto>();
        public IReadOnlyList<TensorInfoDto> Outputs { get; init; } = Array.Empty<TensorInfoDto>();

        public static ModelDetails From(OnnxModel m) => new()
        {
            Id = m.Id,
            Name = m.Name,
            FilePath = m.FilePath,
            Producer = m.ProducerName,
            OpsetVersion = m.OpsetVersion,
            IrVersion = m.IrVersion,
            DocString = m.DocString,
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

        public string RequestExample => new
        {
            inputs = Inputs.ToDictionary(i => i.Name, _ => (object)new[] { 0.0 })
        }.ToJson();

        public static SchemaResponse From(OnnxModel m) => new()
        {
            Inputs = m.Inputs.Select(TensorInfoDto.From).ToList(),
            Outputs = m.Outputs.Select(TensorInfoDto.From).ToList()
        };
    }

    public sealed class PredictResponse
    {
        public string ModelId { get; init; } = string.Empty;
        public long ExecutionTimeMs { get; init; }
        public Dictionary<string, object> Outputs { get; init; } = new();
        public Dictionary<string, IReadOnlyList<long>> OutputShapes { get; init; } = new();

        public static PredictResponse From(OnnxModel model, InferenceResult result)
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
            return value is bool b ? b : System.Convert.ToDouble(value);
        }
    }

    private static string ToJson(this object obj) => JsonSerializer.Serialize(obj, JsonOptions);
}
