using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;

namespace ONNXStudioUI.Controls;

/// <summary>A styled content host with hover/selected border states.</summary>
public class Card : ContentControl
{
}

/// <summary>Indeterminate progress spinner.</summary>
public class Spinner : TemplatedControl
{
}

/// <summary>Bottom status bar of the main window.</summary>
public class StatusBar : TemplatedControl
{
    public static readonly StyledProperty<string> StatusProperty =
        AvaloniaProperty.Register<StatusBar, string>(nameof(Status), "Ready");

    public static readonly StyledProperty<string> InfoProperty =
        AvaloniaProperty.Register<StatusBar, string>(nameof(Info), string.Empty);

    public string Status
    {
        get => GetValue(StatusProperty);
        set => SetValue(StatusProperty, value);
    }

    public string Info
    {
        get => GetValue(InfoProperty);
        set => SetValue(InfoProperty, value);
    }
}
