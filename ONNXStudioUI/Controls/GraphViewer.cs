using System.Globalization;
using System.Windows.Input;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using ONNXStudio.Core.Models;
using ONNXStudioUI.ViewModels.Screens;

namespace ONNXStudioUI.Controls;

/// <summary>Viewport-rendered graph with tensor edges, pointer zoom and drag-to-pan.</summary>
public sealed class GraphViewer : Control
{
    public static readonly StyledProperty<IReadOnlyList<NodeItemViewModel>?> NodesProperty =
        AvaloniaProperty.Register<GraphViewer, IReadOnlyList<NodeItemViewModel>?>(nameof(Nodes));
    public static readonly StyledProperty<ComputationGraph?> GraphProperty =
        AvaloniaProperty.Register<GraphViewer, ComputationGraph?>(nameof(Graph));
    public static readonly StyledProperty<ICommand?> SelectNodeCommandProperty =
        AvaloniaProperty.Register<GraphViewer, ICommand?>(nameof(SelectNodeCommand));
    public IReadOnlyList<NodeItemViewModel>? Nodes { get => GetValue(NodesProperty); set => SetValue(NodesProperty, value); }
    public ComputationGraph? Graph { get => GetValue(GraphProperty); set => SetValue(GraphProperty, value); }
    public ICommand? SelectNodeCommand { get => GetValue(SelectNodeCommandProperty); set => SetValue(SelectNodeCommandProperty, value); }

    private readonly Dictionary<string, Rect> _positions = new();
    private Vector _offset = new(24, 24);
    private double _zoom = 1;
    private Point? _lastPointer;
    private Point _pressPoint;
    private string? _selected;

    public GraphViewer()
    {
        ClipToBounds = true;
        Focusable = true;
    }

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == NodesProperty || change.Property == GraphProperty)
        {
            LayoutGraph();
            InvalidateVisual();
        }
    }

    private void LayoutGraph()
    {
        _positions.Clear();
        _offset = new Vector(24, 24);
        if (Nodes == null || Graph == null) return;
        var ids = Nodes.Select(n => n.Node.Id).ToHashSet();
        var incoming = ids.ToDictionary(id => id, _ => 0);
        var outgoing = ids.ToDictionary(id => id, _ => new List<string>());
        foreach (var edge in Graph.Edges)
        {
            if (!ids.Contains(edge.FromNodeId) || !ids.Contains(edge.ToNodeId)) continue;
            outgoing[edge.FromNodeId].Add(edge.ToNodeId);
            incoming[edge.ToNodeId]++;
        }
        var layers = ids.ToDictionary(id => id, _ => 0);
        var queue = new Queue<string>(Nodes.Where(n => incoming[n.Node.Id] == 0).Select(n => n.Node.Id));
        while (queue.TryDequeue(out var id))
            foreach (var target in outgoing[id])
            {
                layers[target] = Math.Max(layers[target], layers[id] + 1);
                if (--incoming[target] == 0) queue.Enqueue(target);
            }
        var rows = new Dictionary<int, int>();
        foreach (var node in Nodes)
        {
            int layer = layers[node.Node.Id], row = rows.GetValueOrDefault(layer);
            _positions[node.Node.Id] = new Rect(layer * 230, row * 90, 185, 58);
            rows[layer] = row + 1;
        }
    }

    public override void Render(DrawingContext context)
    {
        base.Render(context);
        context.FillRectangle(Brushes.Transparent, new Rect(Bounds.Size));
        if (Nodes == null || Graph == null) return;
        using var transform = context.PushTransform(Matrix.CreateScale(_zoom, _zoom) * Matrix.CreateTranslation(_offset));
        var viewport = new Rect(-_offset.X / _zoom, -_offset.Y / _zoom, Bounds.Width / _zoom, Bounds.Height / _zoom);
        var edgePen = new Pen(Brushes.SlateGray, 1.5);
        foreach (var edge in Graph.Edges)
        {
            if (!_positions.TryGetValue(edge.FromNodeId, out var from) || !_positions.TryGetValue(edge.ToNodeId, out var to)) continue;
            var start = new Point(from.Right, from.Center.Y);
            var end = new Point(to.Left, to.Center.Y);
            if (!viewport.Intersects(new Rect(start, end).Inflate(2))) continue;
            context.DrawLine(edgePen, start, end);
            context.DrawLine(edgePen, end, end + new Vector(-7, -4));
            context.DrawLine(edgePen, end, end + new Vector(-7, 4));
        }
        var background = this.TryFindResource("SurfaceElevatedBrush", out var surface) ? surface as IBrush : null;
        var foreground = this.TryFindResource("TextPrimaryBrush", out var text) ? text as IBrush : null;
        foreach (var node in Nodes)
        {
            var rect = _positions[node.Node.Id];
            if (!viewport.Intersects(rect)) continue;
            context.DrawRectangle(background ?? Brushes.DarkSlateGray, new Pen(node.CategoryBrush, node.Node.Id == _selected ? 4 : 2), rect, 6, 6);
            using var clip = context.PushClip(rect.Deflate(6));
            context.DrawText(Text(node.OpType, 14, foreground ?? Brushes.White), rect.TopLeft + new Vector(10, 7));
            context.DrawText(Text(node.Name, 11, foreground ?? Brushes.LightGray), rect.TopLeft + new Vector(10, 31));
        }
    }

    private static FormattedText Text(string value, double size, IBrush brush) =>
        new(value, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Typeface.Default, size, brush);

    protected override void OnPointerWheelChanged(PointerWheelEventArgs e)
    {
        var point = e.GetPosition(this);
        var next = Math.Clamp(_zoom * Math.Pow(1.15, e.Delta.Y), .1, 4);
        _offset = (Vector)point - ((Vector)point - _offset) * (next / _zoom);
        _zoom = next;
        InvalidateVisual();
        e.Handled = true;
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        _pressPoint = e.GetPosition(this);
        _lastPointer = _pressPoint;
        e.Pointer.Capture(this);
        Focus();
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        var point = e.GetPosition(this);
        if (_lastPointer is Point previous)
        {
            _offset += point - previous;
            _lastPointer = point;
            InvalidateVisual();
        }
        else
        {
            var node = Hit(point);
            ToolTip.SetTip(this, node?.Tooltip);
        }
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        var point = e.GetPosition(this);
        if (((Vector)(point - _pressPoint)).Length < 4 && Hit(point) is { } node)
        {
            _selected = node.Node.Id;
            if (SelectNodeCommand?.CanExecute(node) == true) SelectNodeCommand.Execute(node);
        }
        _lastPointer = null;
        e.Pointer.Capture(null);
        InvalidateVisual();
    }

    protected override void OnPointerCaptureLost(PointerCaptureLostEventArgs e) => _lastPointer = null;

    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e);
        if (e.Key == Key.Home)
        {
            _offset = new Vector(24, 24);
            _zoom = 1;
        }
        else if (Nodes is { Count: > 0 } && e.Key is Key.Left or Key.Right or Key.Up or Key.Down)
        {
            var index = Nodes.ToList().FindIndex(n => n.Node.Id == _selected);
            index = Math.Clamp(index + (e.Key is Key.Left or Key.Up ? -1 : 1), 0, Nodes.Count - 1);
            var node = Nodes[index];
            _selected = node.Node.Id;
            var rect = _positions[_selected];
            _offset = new Vector(24 - rect.X * _zoom, 24 - rect.Y * _zoom);
            if (SelectNodeCommand?.CanExecute(node) == true) SelectNodeCommand.Execute(node);
        }
        else return;
        InvalidateVisual();
        e.Handled = true;
    }

    private NodeItemViewModel? Hit(Point point)
    {
        var local = new Point((point.X - _offset.X) / _zoom, (point.Y - _offset.Y) / _zoom);
        return Nodes?.FirstOrDefault(n => _positions.TryGetValue(n.Node.Id, out var rect) && rect.Contains(local));
    }
}
