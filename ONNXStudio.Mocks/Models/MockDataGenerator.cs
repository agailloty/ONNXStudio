using System;
using System.Collections.Generic;
using System.Linq;

namespace ONNXStudio.Mocks.Models;

/// <summary>
/// Generates mock data for ONNX models and graphs
/// </summary>
public static class MockDataGenerator
{
    private static readonly Random Random = new Random();
    
    public static OnnxModel CreateMockResNet50()
    {
        var model = new OnnxModel
        {
            Id = Guid.NewGuid().ToString(),
            Name = "ResNet50",
            FilePath = "/path/to/resnet50.onnx",
            FileSize = 98 * 1024 * 1024, // 98 MB
            LoadedAt = DateTime.Now.AddMinutes(-5),
            Producer = "PyTorch",
            OpsetVersion = 14,
            IrVersion = "8",
            Graph = CreateResNet50Graph()
        };
        return model;
    }
    
    public static OnnxModel CreateMockBert()
    {
        var model = new OnnxModel
        {
            Id = Guid.NewGuid().ToString(),
            Name = "BERT",
            FilePath = "/path/to/bert.onnx",
            FileSize = 420 * 1024 * 1024, // 420 MB
            LoadedAt = DateTime.Now.AddMinutes(-10),
            Producer = "TensorFlow",
            OpsetVersion = 12,
            IrVersion = "7",
            Graph = CreateBertGraph()
        };
        return model;
    }
    
    public static OnnxModel CreateMockMobileNet()
    {
        var model = new OnnxModel
        {
            Id = Guid.NewGuid().ToString(),
            Name = "MobileNetV2",
            FilePath = "/path/to/mobilenetv2.onnx",
            FileSize = 14 * 1024 * 1024, // 14 MB
            LoadedAt = DateTime.Now.AddMinutes(-2),
            Producer = "PyTorch",
            OpsetVersion = 11,
            IrVersion = "6",
            Graph = CreateMobileNetGraph()
        };
        return model;
    }
    
    public static OnnxModel CreateMockSimpleModel()
    {
        var model = new OnnxModel
        {
            Id = Guid.NewGuid().ToString(),
            Name = "SimpleModel",
            FilePath = "/path/to/simple.onnx",
            FileSize = 1 * 1024 * 1024, // 1 MB
            LoadedAt = DateTime.Now,
            Producer = "ONNX",
            OpsetVersion = 13,
            IrVersion = "8",
            Graph = CreateSimpleGraph()
        };
        return model;
    }
    
    public static ComputationGraph CreateResNet50Graph()
    {
        var graph = new ComputationGraph
        {
            Name = "ResNet50",
            Producer = "PyTorch",
            OpsetVersion = 14,
            IrVersion = "8",
            ParameterCount = 23_000_000
        };
        
        // Input
        graph.Inputs.Add(new GraphTensor
        {
            Name = "input",
            DataType = "float32",
            Shape = new long[] { 1, 3, 224, 224 }
        });
        
        // Conv layers
        for (int i = 0; i < 5; i++)
        {
            graph.Nodes.Add(new GraphNode
            {
                Name = "Conv" + (i + 1),
                OpType = "Conv",
                Domain = "",
                Attributes = new Dictionary<string, object>
                {
                    { "kernel_shape", new int[] { 7, 7 } },
                    { "strides", new int[] { 2, 2 } },
                    { "pads", new int[] { 3, 3, 3, 3 } }
                },
                X = 100 + i * 200,
                Y = 100 + i * 100
            });
        }
        
        // Pool layers
        for (int i = 0; i < 3; i++)
        {
            graph.Nodes.Add(new GraphNode
            {
                Name = "Pool" + (i + 1),
                OpType = "MaxPool",
                Domain = "",
                X = 100 + i * 200,
                Y = 300 + i * 100
            });
        }
        
        // FC layer
        graph.Nodes.Add(new GraphNode
        {
            Name = "fc",
            OpType = "Gemm",
            Domain = "",
            X = 400,
            Y = 500
        });
        
        // Output
        graph.Outputs.Add(new GraphTensor
        {
            Name = "output",
            DataType = "float32",
            Shape = new long[] { 1, 1000 }
        });
        
        // Connect nodes
        for (int i = 0; i < graph.Nodes.Count - 1; i++)
        {
            graph.Nodes[i].OutputIndices = new int[] { i + 1 };
            graph.Nodes[i + 1].InputIndices = new int[] { i };
        }
        
        return graph;
    }
    
    public static ComputationGraph CreateBertGraph()
    {
        var graph = new ComputationGraph
        {
            Name = "BERT",
            Producer = "TensorFlow",
            OpsetVersion = 12,
            IrVersion = "7",
            ParameterCount = 110_000_000
        };
        
        // Inputs
        graph.Inputs.Add(new GraphTensor
        {
            Name = "input_ids",
            DataType = "int64",
            Shape = new long[] { 1, 128 }
        });
        graph.Inputs.Add(new GraphTensor
        {
            Name = "attention_mask",
            DataType = "int64",
            Shape = new long[] { 1, 128 }
        });
        graph.Inputs.Add(new GraphTensor
        {
            Name = "token_type_ids",
            DataType = "int64",
            Shape = new long[] { 1, 128 }
        });
        
        // Embedding
        graph.Nodes.Add(new GraphNode
        {
            Name = "embedding",
            OpType = "Gather",
            X = 100,
            Y = 100
        });
        
        // Transformer layers
        for (int i = 0; i < 12; i++)
        {
            graph.Nodes.Add(new GraphNode
            {
                Name = "EncoderLayer" + i,
                OpType = "Attention",
                X = 300 + (i % 3) * 200,
                Y = 200 + (i / 3) * 150
            });
        }
        
        // Output
        graph.Nodes.Add(new GraphNode
        {
            Name = "pooler",
            OpType = "ReduceMean",
            X = 500,
            Y = 500
        });
        
        graph.Outputs.Add(new GraphTensor
        {
            Name = "output",
            DataType = "float32",
            Shape = new long[] { 1, 768 }
        });
        
        return graph;
    }
    
    public static ComputationGraph CreateMobileNetGraph()
    {
        var graph = new ComputationGraph
        {
            Name = "MobileNetV2",
            Producer = "PyTorch",
            OpsetVersion = 11,
            IrVersion = "6",
            ParameterCount = 3_500_000
        };
        
        // Input
        graph.Inputs.Add(new GraphTensor
        {
            Name = "input",
            DataType = "float32",
            Shape = new long[] { 1, 3, 224, 224 }
        });
        
        // Depthwise separable convolutions
        for (int i = 0; i < 17; i++)
        {
            graph.Nodes.Add(new GraphNode
            {
                Name = "ConvDW" + i,
                OpType = i % 2 == 0 ? "Conv" : "DepthToSpace",
                X = 50 + (i % 4) * 150,
                Y = 100 + (i / 4) * 120
            });
        }
        
        // Output
        graph.Outputs.Add(new GraphTensor
        {
            Name = "output",
            DataType = "float32",
            Shape = new long[] { 1, 1000 }
        });
        
        return graph;
    }
    
    public static ComputationGraph CreateSimpleGraph()
    {
        var graph = new ComputationGraph
        {
            Name = "SimpleModel",
            Producer = "ONNX",
            OpsetVersion = 13,
            IrVersion = "8",
            ParameterCount = 10_000
        };
        
        // Input
        graph.Inputs.Add(new GraphTensor
        {
            Name = "input",
            DataType = "float32",
            Shape = new long[] { 1, 4 }
        });
        
        // Layers
        graph.Nodes.Add(new GraphNode
        {
            Name = "layer1",
            OpType = "MatMul",
            Attributes = new Dictionary<string, object>
            {
                { "weights", new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f } }
            },
            X = 200,
            Y = 100
        });
        
        graph.Nodes.Add(new GraphNode
        {
            Name = "activation",
            OpType = "Relu",
            X = 400,
            Y = 100
        });
        
        graph.Nodes.Add(new GraphNode
        {
            Name = "layer2",
            OpType = "MatMul",
            X = 600,
            Y = 100
        });
        
        // Output
        graph.Outputs.Add(new GraphTensor
        {
            Name = "output",
            DataType = "float32",
            Shape = new long[] { 1, 2 }
        });
        
        // Connect nodes
        graph.Nodes[0].InputIndices = new int[] { 0 };
        graph.Nodes[0].OutputIndices = new int[] { 1 };
        graph.Nodes[1].InputIndices = new int[] { 1 };
        graph.Nodes[1].OutputIndices = new int[] { 2 };
        graph.Nodes[2].InputIndices = new int[] { 2 };
        graph.Nodes[2].OutputIndices = new int[] { 0 };
        
        return graph;
    }
    
    public static List<InferenceInput> CreateMockInferenceInputs(OnnxModel model)
    {
        var inputs = new List<InferenceInput>();

        if (model.Name.Contains("housing"))
        {
            // One numeric slider per California Housing feature
            var features = new (string name, string label, double min, double max, double step, double def)[]
            {
                ("MedInc", "Median income (x10k USD)", 0.5, 15.0, 0.1, 5.41),
                ("HouseAge", "Median house age (years)", 1.0, 52.0, 1.0, 25.0),
                ("AveRooms", "Average rooms / household", 1.0, 20.0, 0.1, 6.28),
                ("AveBedrms", "Average bedrooms / household", 0.5, 5.0, 0.05, 1.07),
                ("Population", "Block population", 100.0, 10000.0, 50.0, 1425.0),
                ("AveOccup", "Average occupants / household", 1.0, 10.0, 0.1, 3.07),
                ("Latitude", "Latitude", 32.0, 42.0, 0.1, 35.63),
                ("Longitude", "Longitude", -125.0, -114.0, 0.1, -119.57)
            };

            foreach (var (name, label, min, max, step, def) in features)
            {
                inputs.Add(new InferenceInput
                {
                    Name = name,
                    DisplayName = label,
                    Type = "float32",
                    Shape = new long[] { 1 },
                    InputType = "Number",
                    Min = min,
                    Max = max,
                    Step = step,
                    Value = def,
                    IsRequired = true
                });
            }

            return inputs;
        }

        if (model.Name.Contains("ResNet") || model.Name.Contains("MobileNet"))
        {
            inputs.Add(new InferenceInput
            {
                Name = "input",
                DisplayName = "Input Image",
                Type = "float32",
                Shape = new long[] { 1, 3, 224, 224 },
                InputType = "Image",
                IsRequired = true
            });
        }
        else if (model.Name.Contains("BERT"))
        {
            inputs.Add(new InferenceInput
            {
                Name = "input_ids",
                DisplayName = "Input IDs",
                Type = "int64",
                Shape = new long[] { 1, 128 },
                InputType = "Vector",
                VectorValues = string.Join(", ", Enumerable.Range(0, 128).Select(_ => Random.Next(0, 1000))),
                IsRequired = true
            });
            inputs.Add(new InferenceInput
            {
                Name = "attention_mask",
                DisplayName = "Attention Mask",
                Type = "int64",
                Shape = new long[] { 1, 128 },
                InputType = "Vector",
                VectorValues = string.Join(", ", Enumerable.Repeat(1, 128)),
                IsRequired = true
            });
        }
        else
        {
            inputs.Add(new InferenceInput
            {
                Name = "input",
                DisplayName = "Input",
                Type = "float32",
                Shape = new long[] { 1, 4 },
                InputType = "Vector",
                VectorValues = "0.1, 0.2, 0.3, 0.4",
                IsRequired = true
            });
        }
        
        return inputs;
    }
    
    public static InferenceResult CreateMockInferenceResult(OnnxModel model, List<InferenceInput> inputs)
    {
        var result = new InferenceResult
        {
            ModelId = model.Id,
            ModelName = model.Name,
            ExecutionTime = DateTime.Now.AddMilliseconds(-450),
            IsSuccess = true
        };
        
        if (model.Name.Contains("housing"))
        {
            // Regression result: predicted median house value
            result.Outputs.Add(new InferenceOutput
            {
                Name = "prediction",
                DisplayName = "Predicted MedHouseVal",
                Type = "float32",
                Shape = new long[] { 1 },
                Value = 2.053f,
                DisplayValue = "$205,300 (MedHouseVal = 2.05)"
            });

            return result;
        }

        if (model.Name.Contains("ResNet") || model.Name.Contains("MobileNet"))
        {
            // Classification result
            var output = new InferenceOutput
            {
                Name = "output",
                DisplayName = "Classification Output",
                Type = "float32",
                Shape = new long[] { 1, 1000 },
                Value = new float[1000]
            };
            
            // Generate mock probabilities
            var probabilities = new float[1000];
            var random = new Random(42); // Seed for reproducibility
            for (int i = 0; i < 1000; i++)
            {
                probabilities[i] = (float)random.NextDouble();
            }
            
            // Normalize to sum to 1
            var sum = probabilities.Sum();
            for (int i = 0; i < probabilities.Length; i++)
            {
                probabilities[i] /= (float)sum;
            }
            
            output.Value = probabilities;
            
            // Add top 5 classifications
            var indices = Enumerable.Range(0, 1000)
                .Select(i => new { Index = i, Value = probabilities[i] })
                .OrderByDescending(x => x.Value)
                .Take(5)
                .ToList();
            
            var labels = new[] { "Cat", "Dog", "Bird", "Car", "Fish", "Tree", "Person", "House", "Sky", "Water" };
            foreach (var idx in indices)
            {
                output.ClassificationResults.Add(new ClassificationResult
                {
                    Label = labels[idx.Index % labels.Length],
                    Confidence = (double)idx.Value,
                    Index = idx.Index
                });
            }
            
            result.Outputs.Add(output);
        }
        else if (model.Name.Contains("BERT"))
        {
            // Embedding result
            var output = new InferenceOutput
            {
                Name = "output",
                DisplayName = "Embedding",
                Type = "float32",
                Shape = new long[] { 1, 768 },
                Value = new float[768]
            };
            
            var random = new Random(42);
            for (int i = 0; i < 768; i++)
            {
                ((float[])output.Value)[i] = (float)(random.NextDouble() * 2 - 1);
            }
            
            result.Outputs.Add(output);
        }
        else
        {
            // Simple output
            var output = new InferenceOutput
            {
                Name = "output",
                DisplayName = "Output",
                Type = "float32",
                Shape = new long[] { 1, 2 },
                Value = new float[] { 0.85f, 0.15f }
            };
            result.Outputs.Add(output);
        }
        
        return result;
    }
    
    public static List<ApiEndpoint> CreateMockApiEndpoints(OnnxModel model)
    {
        return new List<ApiEndpoint>
        {
            new ApiEndpoint
            {
                Method = "GET",
                Path = "/models",
                Description = "List all loaded models",
                ModelId = model.Id,
                ModelName = model.Name
            },
            new ApiEndpoint
            {
                Method = "GET",
                Path = "/models/" + model.Name,
                Description = "Get model details",
                ModelId = model.Id,
                ModelName = model.Name
            },
            new ApiEndpoint
            {
                Method = "POST",
                Path = "/models/" + model.Name + "/predict",
                Description = "Run inference on the model",
                ModelId = model.Id,
                ModelName = model.Name,
                RequestExample = "{\"input\": {\"data\": [0.1, 0.2, ...]}}",
                ResponseExample = "{\"output\": [0.1, 0.2, ...]}"
            },
            new ApiEndpoint
            {
                Method = "GET",
                Path = "/models/" + model.Name + "/schema",
                Description = "Get model input/output schema",
                ModelId = model.Id,
                ModelName = model.Name
            }
        };
    }

    // ====================================================================
    // California Housing - scikit-learn pipeline (StandardScaler + GBR)
    // converted to ONNX with skl2onnx. Regression on MedHouseVal ($100k).
    // ====================================================================

    public static string[] HousingFeatures { get; } =
    {
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    };

    public static OnnxModel CreateMockCaliforniaHousingPipeline()
    {
        var model = new OnnxModel
        {
            Id = Guid.NewGuid().ToString(),
            Name = "pipeline_housing",
            FilePath = "/models/pipeline_housing.onnx",
            FileSize = 486_000, // ~475 KB
            LoadedAt = DateTime.Now.AddMinutes(-1),
            Producer = "skl2onnx (scikit-learn 1.4)",
            OpsetVersion = 15,
            IrVersion = "9",
            DocString = "sklearn Pipeline([StandardScaler, GradientBoostingRegressor]) exported to ONNX",
            Graph = CreateHousingPipelineGraph()
        };
        return model;
    }

    private static ComputationGraph CreateHousingPipelineGraph()
    {
        var graph = new ComputationGraph
        {
            Name = "Pipeline(StandardScaler, GradientBoostingRegressor)",
            Producer = "skl2onnx (scikit-learn 1.4)",
            OpsetVersion = 15,
            IrVersion = "9",
            ParameterCount = 48_214,
            Inputs = new List<GraphTensor>
            {
                new()
                {
                    Name = "float_input",
                    DataType = "float32",
                    Shape = new long[] { 1, 8 }
                }
            },
            Outputs = new List<GraphTensor>
            {
                new()
                {
                    Name = "prediction",
                    DataType = "float32",
                    Shape = new long[] { 1 }
                }
            },
            Nodes = new List<GraphNode>
            {
                new()
                {
                    Name = "StandardScaler",
                    OpType = "Scaler",
                    Attributes = new Dictionary<string, object>
                    {
                        { "offset", "[2.129, 28.6, 4.34, 1.09, 1425.5, 3.07, 35.63, -119.57]" },
                        { "scale", "[1.90, 12.6, 10.3, 0.47, 1132.5, 3.99, 2.14, 2.00]" }
                    },
                    Inputs = new List<GraphTensor>
                    {
                        new() { Name = "float_input", DataType = "float32", Shape = new long[] { 1, 8 } }
                    },
                    Outputs = new List<GraphTensor>
                    {
                        new() { Name = "scaled_input", DataType = "float32", Shape = new long[] { 1, 8 } }
                    }
                },
                new()
                {
                    Name = "GradientBoostingRegressor",
                    OpType = "TreeEnsembleRegressor",
                    Attributes = new Dictionary<string, object>
                    {
                        { "n_estimators", 300 },
                        { "max_depth", 3 },
                        { "n_targets", 1 },
                        { "base_values", 2.07 },
                        { "post_transform", "regression" }
                    },
                    Inputs = new List<GraphTensor>
                    {
                        new() { Name = "scaled_input", DataType = "float32", Shape = new long[] { 1, 8 } }
                    },
                    Outputs = new List<GraphTensor>
                    {
                        new() { Name = "prediction", DataType = "float32", Shape = new long[] { 1 } }
                    }
                }
            }
        };

        return graph;
    }

    /// <summary>
    /// Graph-free structure of a model: pipeline steps with parameters, or an
    /// aggregate of graph node types for non-pipeline models.
    /// </summary>
    public static List<PipelineComponent> CreateMockPipelineStructure(OnnxModel model)
    {
        if (model.Name.Contains("housing"))
        {
            return new List<PipelineComponent>
            {
                new PipelineComponent
                {
                    StepName = "pipeline",
                    Name = "Pipeline",
                    Kind = "pipeline",
                    Icon = "🔧",
                    Description = "sklearn.pipeline.Pipeline - chain of transforms and a final regressor",
                    Parameters = new List<ParameterEntry>
                    {
                        new() { Key = "steps", Value = "[('scaler', StandardScaler), ('regressor', GradientBoostingRegressor)]", ValueType = "array" },
                        new() { Key = "memory", Value = "None" },
                        new() { Key = "verbose", Value = "False", ValueType = "bool" }
                    },
                    Children = new List<PipelineComponent>
                    {
                        new PipelineComponent
                        {
                            StepName = "scaler",
                            Name = "StandardScaler",
                            Kind = "preprocessing",
                            Icon = "📊",
                            Description = "Standardize features by removing the mean and scaling to unit variance",
                            Parameters = new List<ParameterEntry>
                            {
                                new() { Key = "with_mean", Value = "True", ValueType = "bool" },
                                new() { Key = "with_std", Value = "True", ValueType = "bool" },
                                new() { Key = "copy", Value = "True", ValueType = "bool" },
                                new() { Key = "n_features_in_", Value = "8", ValueType = "number" },
                                new() { Key = "n_samples_seen_", Value = "20640", ValueType = "number" },
                                new() { Key = "mean_", Value = "[2.129, 28.6, 4.34, 1.09, 1425.5, 3.07, 35.63, -119.57]", ValueType = "array" },
                                new() { Key = "scale_", Value = "[1.90, 12.6, 10.3, 0.47, 1132.5, 3.99, 2.14, 2.00]", ValueType = "array" }
                            }
                        },
                        new PipelineComponent
                        {
                            StepName = "regressor",
                            Name = "GradientBoostingRegressor",
                            Kind = "regressor",
                            Icon = "🌿",
                            Description = "Gradient Boosting for regression, trained on California Housing (MedHouseVal)",
                            Parameters = new List<ParameterEntry>
                            {
                                new() { Key = "loss", Value = "squared_error" },
                                new() { Key = "learning_rate", Value = "0.1", ValueType = "number" },
                                new() { Key = "n_estimators", Value = "300", ValueType = "number" },
                                new() { Key = "subsample", Value = "1.0", ValueType = "number" },
                                new() { Key = "criterion", Value = "friedman_mse" },
                                new() { Key = "min_samples_split", Value = "2", ValueType = "number" },
                                new() { Key = "min_samples_leaf", Value = "1", ValueType = "number" },
                                new() { Key = "max_depth", Value = "3", ValueType = "number" },
                                new() { Key = "max_features", Value = "None" },
                                new() { Key = "validation_fraction", Value = "0.1", ValueType = "number" }
                            }
                        }
                    }
                }
            };
        }

        // Fallback: aggregate graph nodes by OpType (intuitive for CNN/transformer models)
        var groups = model.Graph.Nodes
            .GroupBy(n => n.OpType)
            .Select(g => new PipelineComponent
            {
                StepName = g.Key.ToLowerInvariant(),
                Name = g.Key,
                Kind = "layer-group",
                Icon = "▦",
                Description = g.Count() + " node(s) of type " + g.Key,
                Parameters = g.First().Attributes
                    .Select(a => new ParameterEntry { Key = a.Key, Value = a.Value?.ToString() ?? "", ValueType = "str" })
                    .ToList()
            })
            .ToList();

        return new List<PipelineComponent>
        {
            new PipelineComponent
            {
                StepName = "graph",
                Name = "Graph",
                Kind = "model",
                Icon = "🧩",
                Description = "Computation graph - " + model.Graph.NodeCount + " nodes aggregated by operator type",
                Children = groups
            }
        };
    }

    /// <summary>
    /// Feature importances (only meaningful for the housing pipeline in this mock).
    /// </summary>
    public static List<FeatureImportance> CreateMockFeatureImportances(OnnxModel model)
    {
        if (!model.Name.Contains("housing"))
        {
            return new List<FeatureImportance>();
        }

        return new List<FeatureImportance>
        {
            new() { Feature = "MedInc", Importance = 0.582 },
            new() { Feature = "AveOccup", Importance = 0.134 },
            new() { Feature = "Latitude", Importance = 0.118 },
            new() { Feature = "Longitude", Importance = 0.091 },
            new() { Feature = "HouseAge", Importance = 0.032 },
            new() { Feature = "AveRooms", Importance = 0.021 },
            new() { Feature = "Population", Importance = 0.014 },
            new() { Feature = "AveBedrms", Importance = 0.008 }
        };
    }

    /// <summary>
    /// Default payload field mappings for the API endpoint of a model.
    /// </summary>
    public static List<ApiFieldMapping> CreateMockApiFieldMappings(OnnxModel model)
    {
        var mappings = new List<ApiFieldMapping>();

        if (model.Name.Contains("housing"))
        {
            var descriptions = new Dictionary<string, string>
            {
                { "MedInc", "Median income in block group (x10k USD)" },
                { "HouseAge", "Median house age in block group (years)" },
                { "AveRooms", "Average number of rooms per household" },
                { "AveBedrms", "Average number of bedrooms per household" },
                { "Population", "Block group population" },
                { "AveOccup", "Average number of household members" },
                { "Latitude", "Block group latitude" },
                { "Longitude", "Block group longitude" }
            };
            var samples = new Dictionary<string, string>
            {
                { "MedInc", "5.41" }, { "HouseAge", "25.0" }, { "AveRooms", "6.28" },
                { "AveBedrms", "1.07" }, { "Population", "1425" }, { "AveOccup", "3.07" },
                { "Latitude", "35.63" }, { "Longitude", "-119.57" }
            };

            for (int i = 0; i < HousingFeatures.Length; i++)
            {
                var feature = HousingFeatures[i];
                mappings.Add(new ApiFieldMapping
                {
                    SourceName = "float_input",
                    SourceType = "float32",
                    SourceColumn = i,
                    JsonName = feature,
                    JsonType = "number",
                    IsRequired = true,
                    SampleValue = samples[feature],
                    Description = descriptions[feature]
                });
            }

            return mappings;
        }

        // Generic fallback: one mapping per graph input tensor
        foreach (var input in model.Graph.Inputs)
        {
            mappings.Add(new ApiFieldMapping
            {
                SourceName = input.Name,
                SourceType = input.DataType,
                JsonName = input.Name,
                JsonType = input.DataType.Contains("64") || input.DataType.Contains("int") ? "array" : "array",
                IsRequired = true,
                SampleValue = "[0.1, 0.2, 0.3]",
                Description = "Model input tensor " + input.Name
            });
        }

        return mappings;
    }
}
