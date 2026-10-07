using ONNXStudioUI.Services;
using Xunit;

namespace ONNXStudio.UI.Tests;

public class SettingsTests
{
    [Fact]
    public void SettingsSurviveRestartAndCorruptFilesFallBackToDefaults()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        try
        {
            var store = new SettingsStore(path);
            store.Save(AppTheme.System, 9001);
            Assert.Equal((AppTheme.System, 9001), new SettingsStore(path).Load());
            File.WriteAllText(path, "{broken");
            Assert.Equal((AppTheme.Dark, 5000), store.Load());
        }
        finally { File.Delete(path); }
    }
}
