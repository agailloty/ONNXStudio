using System.Text;

namespace ONNXStudio.Core.Utilities;

/// <summary>
/// Generic protobuf wire-format reader (varint / fixed64 / length-delimited /
/// fixed32) used to walk ONNX ModelProto messages without any external
/// protobuf dependency. AOT friendly: no reflection, no dynamic code.
/// </summary>
public ref struct ProtoReader
{
    private readonly ReadOnlySpan<byte> _buffer;

    public ProtoReader(ReadOnlySpan<byte> buffer)
    {
        _buffer = buffer;
    }

    public bool HasMore => !_buffer.IsEmpty;

    /// <summary>Field header: tag number and wire type.</summary>
    public (int FieldNumber, int WireType) ReadFieldHeader()
    {
        var tag = ReadVarInt();
        return ((int)(tag >> 3), (int)(tag & 0x7));
    }

    public ulong ReadVarInt()
    {
        ulong value = 0;
        int shift = 0;
        int index = 0;
        while (index < _buffer.Length)
        {
            byte b = _buffer[index];
            value |= (ulong)(b & 0x7F) << shift;
            index++;
            if ((b & 0x80) == 0)
            {
                // Consume the bytes we read
                this = new ProtoReader(_buffer[index..]);
                return value;
            }
            shift += 7;
            if (shift >= 70)
            {
                throw new InvalidDataException("Protobuf varint too long.");
            }
        }
        throw new InvalidDataException("Truncated protobuf varint.");
    }

    /// <summary>Length-delimited payload (bytes as span).</summary>
    public ReadOnlySpan<byte> ReadLengthDelimited()
    {
        long length = (long)ReadVarInt();
        if (length < 0 || length > _buffer.Length)
        {
            throw new InvalidDataException("Protobuf length out of bounds.");
        }
        var payload = _buffer[..(int)length];
        this = new ProtoReader(_buffer[(int)length..]);
        return payload;
    }

    public string ReadString() => Encoding.UTF8.GetString(ReadLengthDelimited());

    /// <summary>Reads a fixed32 (4 bytes, little-endian).</summary>
    public uint ReadFixed32()
    {
        if (_buffer.Length < 4)
        {
            throw new InvalidDataException("Truncated fixed32.");
        }
        var value = BitConverter.ToUInt32(_buffer[..4]);
        this = new ProtoReader(_buffer[4..]);
        return value;
    }

    /// <summary>Reads a fixed64 (8 bytes, little-endian).</summary>
    public ulong ReadFixed64()
    {
        if (_buffer.Length < 8)
        {
            throw new InvalidDataException("Truncated fixed64.");
        }
        var value = BitConverter.ToUInt64(_buffer[..8]);
        this = new ProtoReader(_buffer[8..]);
        return value;
    }

    /// <summary>
    /// Reads a packed repeated varint field (int64 dims, float_data uses fixed32...).
    /// </summary>
    public List<long> ReadPackedVarInts()
    {
        var payload = ReadLengthDelimited();
        var reader = new ProtoReader(payload);
        var values = new List<long>();
        while (reader.HasMore)
        {
            values.Add((long)reader.ReadVarInt());
        }
        return values;
    }

    public List<float> ReadPackedFloats()
    {
        var payload = ReadLengthDelimited();
        var values = new List<float>(payload.Length / 4);
        for (int i = 0; i + 4 <= payload.Length; i += 4)
        {
            values.Add(BitConverter.ToSingle(payload[i..(i + 4)]));
        }
        return values;
    }

    public void SkipField(int wireType)
    {
        switch (wireType)
        {
            case 0: ReadVarInt(); break;
            case 1: this = new ProtoReader(_buffer[8..]); break;
            case 2: ReadLengthDelimited(); break;
            case 5: this = new ProtoReader(_buffer[4..]); break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wireType}.");
        }
    }
}

/// <summary>
/// Typed view over the subset of the ONNX ModelProto message needed by the
/// studio: metadata, opset imports, graph nodes with attributes, and the
/// input/output value infos (with dynamic dimensions).
/// </summary>
public static class OnnxProtoParser
{
    // ModelProto fields
    private const int FieldIrVersion = 1;
    private const int FieldProducerName = 2;
    private const int FieldProducerVersion = 3;
    private const int FieldDomain = 4;
    private const int FieldModelVersion = 5;
    private const int FieldDocString = 6;
    private const int FieldGraph = 7;
    private const int FieldOpsetImport = 8;
    private const int FieldMetadataProps = 14;

    // GraphProto fields
    private const int GraphNode = 1;
    private const int GraphName = 2;
    private const int GraphInitializer = 5;
    private const int GraphInput = 11;
    private const int GraphOutput = 12;
    private const int GraphValueInfo = 13;

    // NodeProto fields
    private const int NodeInput = 1;
    private const int NodeOutput = 2;
    private const int NodeName = 3;
    private const int NodeOpType = 4;
    private const int NodeAttribute = 5;
    private const int NodeDocString = 6;
    private const int NodeDomain = 7;

    // AttributeProto fields
    private const int AttrName = 1;
    private const int AttrFloat = 2;
    private const int AttrInt = 3;
    private const int AttrString = 4;
    private const int AttrTensor = 5;
    private const int AttrFloats = 7;
    private const int AttrInts = 8;
    private const int AttrStrings = 9;
    private const int AttrType = 20;

    // ValueInfoProto fields
    private const int ValueInfoName = 1;
    private const int ValueInfoType = 2;

    // TypeProto fields
    private const int TypeTensor = 1;

    // TypeProto.Tensor fields
    private const int TensorElemType = 1;
    private const int TensorShape = 2;

    // TensorShapeProto fields
    private const int ShapeDim = 1;

    // Dimension fields
    private const int DimValue = 1;
    private const int DimParam = 2;

    // TensorProto fields
    private const int TensorDims = 1;
    private const int TensorDataType = 2;
    private const int TensorName = 8;

    // OperatorSetIdProto fields
    private const int OpsetDomain = 1;
    private const int OpsetVersion = 2;

    public sealed class RawModel
    {
        public long IrVersion;
        public string ProducerName = string.Empty;
        public string ProducerVersion = string.Empty;
        public string Domain = string.Empty;
        public long ModelVersion;
        public string DocString = string.Empty;
        public long OpsetVersion;
        public Dictionary<string, string> Metadata = new();
        public List<RawNode> Nodes = new();
        public List<RawValueInfo> Inputs = new();
        public List<RawValueInfo> Outputs = new();
        public List<RawTensor> Initializers = new();
    }

    public sealed class RawNode
    {
        public string Name = string.Empty;
        public string OpType = string.Empty;
        public string Domain = string.Empty;
        public List<string> Inputs = new();
        public List<string> Outputs = new();
        public Dictionary<string, object> Attributes = new();
    }

    public sealed class RawValueInfo
    {
        public string Name = string.Empty;
        public int ElementType;
        public List<long?> Shape = new();
        public string Description = string.Empty;
    }

    public sealed class RawTensor
    {
        public string Name = string.Empty;
        public List<long> Dims = new();
        public int DataType;
    }

    /// <summary>
    /// Quick sanity check: ONNX files are ModelProto protobufs that normally
    /// start with field 1 (ir_version) => tag byte 0x08.
    /// </summary>
    public static bool LooksLikeOnnxModel(ReadOnlySpan<byte> bytes)
        => !bytes.IsEmpty && bytes[0] == 0x08;

    public static RawModel Parse(ReadOnlySpan<byte> bytes)
    {
        var model = new RawModel();
        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            switch (field)
            {
                case FieldIrVersion when wire == 0:
                    model.IrVersion = (long)reader.ReadVarInt();
                    break;
                case FieldProducerName when wire == 2:
                    model.ProducerName = reader.ReadString();
                    break;
                case FieldProducerVersion when wire == 2:
                    model.ProducerVersion = reader.ReadString();
                    break;
                case FieldDomain when wire == 2:
                    model.Domain = reader.ReadString();
                    break;
                case FieldModelVersion when wire == 0:
                    model.ModelVersion = (long)reader.ReadVarInt();
                    break;
                case FieldDocString when wire == 2:
                    model.DocString = reader.ReadString();
                    break;
                case FieldGraph when wire == 2:
                    ParseGraph(reader.ReadLengthDelimited(), model);
                    break;
                case FieldMetadataProps when wire == 2:
                    var (key, value) = ParseStringEntry(reader.ReadLengthDelimited());
                    if (key.Length > 0) model.Metadata[key] = value;
                    break;
                case FieldOpsetImport when wire == 2:
                    var opset = ParseOpset(reader.ReadLengthDelimited());
                    // The default (ONNX) domain opset is the relevant one
                    if (opset.Domain.Length == 0)
                    {
                        model.OpsetVersion = opset.Version;
                    }
                    break;
                default:
                    reader.SkipField(wire);
                    break;
            }
        }
        return model;
    }

    /// <summary>StringStringEntryProto: key = 1, value = 2.</summary>
    private static (string Key, string Value) ParseStringEntry(ReadOnlySpan<byte> bytes)
    {
        string key = string.Empty, value = string.Empty;
        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            switch (field)
            {
                case 1 when wire == 2: key = reader.ReadString(); break;
                case 2 when wire == 2: value = reader.ReadString(); break;
                default: reader.SkipField(wire); break;
            }
        }
        return (key, value);
    }

    private static (string Domain, long Version) ParseOpset(ReadOnlySpan<byte> bytes)
    {
        string domain = string.Empty;
        long version = 0;
        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            switch (field)
            {
                case OpsetDomain when wire == 2: domain = reader.ReadString(); break;
                case OpsetVersion when wire == 0: version = (long)reader.ReadVarInt(); break;
                default: reader.SkipField(wire); break;
            }
        }
        return (domain, version);
    }

    private static void ParseGraph(ReadOnlySpan<byte> bytes, RawModel model)
    {
        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            switch (field)
            {
                case GraphNode when wire == 2:
                    model.Nodes.Add(ParseNode(reader.ReadLengthDelimited()));
                    break;
                case GraphInput when wire == 2:
                    model.Inputs.Add(ParseValueInfo(reader.ReadLengthDelimited()));
                    break;
                case GraphOutput when wire == 2:
                    model.Outputs.Add(ParseValueInfo(reader.ReadLengthDelimited()));
                    break;
                case GraphInitializer when wire == 2:
                    model.Initializers.Add(ParseTensor(reader.ReadLengthDelimited()));
                    break;
                default:
                    reader.SkipField(wire);
                    break;
            }
        }
    }

    private static RawNode ParseNode(ReadOnlySpan<byte> bytes)
    {
        var node = new RawNode();
        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            switch (field)
            {
                case NodeInput when wire == 2: node.Inputs.Add(reader.ReadString()); break;
                case NodeOutput when wire == 2: node.Outputs.Add(reader.ReadString()); break;
                case NodeName when wire == 2: node.Name = reader.ReadString(); break;
                case NodeOpType when wire == 2: node.OpType = reader.ReadString(); break;
                case NodeDomain when wire == 2: node.Domain = reader.ReadString(); break;
                case NodeAttribute when wire == 2: ParseAttribute(reader.ReadLengthDelimited(), node); break;
                default: reader.SkipField(wire); break;
            }
        }
        return node;
    }

    private static void ParseAttribute(ReadOnlySpan<byte> bytes, RawNode node)
    {
        string name = string.Empty;
        int type = 0;
        float f = 0;
        long i = 0;
        string s = string.Empty;
        var floats = new List<float>();
        var ints = new List<long>();

        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            switch (field)
            {
                case AttrName when wire == 2: name = reader.ReadString(); break;
                case AttrType when wire == 0: type = (int)reader.ReadVarInt(); break;
                case AttrFloat when wire == 5:
                    f = BitConverter.Int32BitsToSingle(unchecked((int)reader.ReadFixed32()));
                    break;
                case AttrFloat when wire == 1:
                    f = (float)BitConverter.Int64BitsToDouble(unchecked((long)reader.ReadFixed64()));
                    break;
                case AttrInt when wire == 0: i = (long)reader.ReadVarInt(); break;
                case AttrString when wire == 2: s = reader.ReadString(); break;
                case AttrFloats when wire == 2: floats = reader.ReadPackedFloats(); break;
                case AttrFloats when wire == 5: floats.Add(BitConverter.Int32BitsToSingle(unchecked((int)reader.ReadFixed32()))); break;
                case AttrInts when wire == 2: ints = reader.ReadPackedVarInts(); break;
                case AttrInts when wire == 0: ints.Add((long)reader.ReadVarInt()); break;
                default: reader.SkipField(wire); break;
            }
        }

        object value = type switch
        {
            1 => f,               // FLOAT
            2 => i,               // INT
            3 => s,               // STRING
            4 => s,               // TENSOR (name only)
            5 => i,               // GRAPH (name only)
            6 => floats,          // FLOATS
            7 => ints,            // INTS
            8 => new List<string> { s }, // STRINGS (single)
            _ => s
        };
        node.Attributes[name] = value;
    }

    private static RawValueInfo ParseValueInfo(ReadOnlySpan<byte> bytes)
    {
        var info = new RawValueInfo();
        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            switch (field)
            {
                case ValueInfoName when wire == 2:
                    info.Name = reader.ReadString();
                    break;
                case ValueInfoType when wire == 2:
                    ParseType(reader.ReadLengthDelimited(), info);
                    break;
                default:
                    reader.SkipField(wire);
                    break;
            }
        }
        return info;
    }

    private static void ParseType(ReadOnlySpan<byte> bytes, RawValueInfo info)
    {
        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            if (field == TypeTensor && wire == 2)
            {
                ParseTensorType(reader.ReadLengthDelimited(), info);
            }
            else
            {
                reader.SkipField(wire);
            }
        }
    }

    private static void ParseTensorType(ReadOnlySpan<byte> bytes, RawValueInfo info)
    {
        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            switch (field)
            {
                case TensorElemType when wire == 0:
                    info.ElementType = (int)reader.ReadVarInt();
                    break;
                case TensorShape when wire == 2:
                    ParseShape(reader.ReadLengthDelimited(), info);
                    break;
                default:
                    reader.SkipField(wire);
                    break;
            }
        }
    }

    private static void ParseShape(ReadOnlySpan<byte> bytes, RawValueInfo info)
    {
        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            if (field == ShapeDim && wire == 2)
            {
                info.Shape.Add(ParseDimension(reader.ReadLengthDelimited()));
            }
            else
            {
                reader.SkipField(wire);
            }
        }
    }

    private static long? ParseDimension(ReadOnlySpan<byte> bytes)
    {
        long? value = null;
        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            switch (field)
            {
                case DimValue when wire == 0: value = (long)reader.ReadVarInt(); break;
                case DimParam when wire == 2: reader.ReadString(); break; // dynamic dimension
                default: reader.SkipField(wire); break;
            }
        }
        return value;
    }

    private static RawTensor ParseTensor(ReadOnlySpan<byte> bytes)
    {
        var tensor = new RawTensor();
        var reader = new ProtoReader(bytes);
        while (reader.HasMore)
        {
            var (field, wire) = reader.ReadFieldHeader();
            switch (field)
            {
                case TensorDims when wire == 2:
                    tensor.Dims.AddRange(reader.ReadPackedVarInts());
                    break;
                case TensorDims when wire == 0:
                    tensor.Dims.Add((long)reader.ReadVarInt());
                    break;
                case TensorDataType when wire == 0:
                    tensor.DataType = (int)reader.ReadVarInt();
                    break;
                case TensorName when wire == 2:
                    tensor.Name = reader.ReadString();
                    break;
                default:
                    reader.SkipField(wire);
                    break;
            }
        }
        return tensor;
    }
}
