using System.IO;
using System.Text.Json;
using System.Windows;

namespace Dubly.Launcher;

public partial class App : Application
{
    Mutex? instance;
    protected override async void OnStartup(StartupEventArgs e)
    {
        System.Globalization.CultureInfo.DefaultThreadCurrentUICulture = System.Globalization.CultureInfo.GetCultureInfo("en-US");
        base.OnStartup(e);
        if (e.Args.Length == 2 && e.Args[0] == "--preview") { new MainWindow().RenderPreview(e.Args[1]); Shutdown(); return; }
        if (e.Args.Length == 3 && e.Args[0] == "--smoke-test")
        {
            var path = e.Args[1]; var port = int.Parse(e.Args[2]); var root = Path.Combine(Path.GetTempPath(), "dubly-test-" + Guid.NewGuid().ToString("N"));
            try
            {
                var store = new SettingsStore(root); store.Save(port, "test-local-placeholder-key");
                var encrypted = File.ReadAllText(Path.Combine(root, "settings.json"));
                if (encrypted.Contains("test-local-placeholder-key") || store.Load().Key != "test-local-placeholder-key") throw new Exception("Credential encryption test failed");
                using var controller = new ServerController(AppContext.BaseDirectory, root);
                await controller.StartAsync(port, store.Load().Key); var started = controller.IsRunning;
                using var conflict = new ServerController(AppContext.BaseDirectory, root); var conflictBlocked = false;
                try { await conflict.StartAsync(port, "test-other-placeholder-key"); } catch (InvalidOperationException) { conflictBlocked = true; }
                await controller.StopAsync(); var stopped = !controller.IsRunning;
                File.WriteAllText(path, JsonSerializer.Serialize(new { started, stopped, conflictBlocked, credentialsEncrypted = true }));
                Shutdown(started && stopped && conflictBlocked ? 0 : 1);
            }
            catch (Exception error) { File.WriteAllText(path, JsonSerializer.Serialize(new { error = error.Message })); Shutdown(1); }
            return;
        }
        instance = new Mutex(true, @"Local\Dubly.Server.Launcher", out var first);
        if (!first) { MessageBox.Show("Dubly 서버 실행기가 이미 켜져 있습니다.", "Dubly"); Shutdown(); return; }
        MainWindow = new MainWindow(); MainWindow.Show();
    }
    protected override void OnExit(ExitEventArgs e) { instance?.Dispose(); base.OnExit(e); }
}
