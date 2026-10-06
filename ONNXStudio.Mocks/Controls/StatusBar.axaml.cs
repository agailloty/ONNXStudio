using Avalonia;
using Avalonia.Controls.Primitives;

namespace ONNXStudio.Mocks.Controls;

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
