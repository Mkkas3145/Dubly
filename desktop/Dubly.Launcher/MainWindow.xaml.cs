using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace Dubly.Launcher;

public partial class MainWindow : Window
{
    readonly SettingsStore settings = new(); readonly ServerController server; readonly DispatcherTimer monitor;
    bool busy, allowClose, wasRunning;
    public MainWindow()
    {
        InitializeComponent(); server = new(AppContext.BaseDirectory, Path.Combine(settings.DirectoryPath, "logs"));
        var saved = settings.Load(); PortBox.Text = saved.Port.ToString(); KeyBox.Password = saved.Key;
        monitor = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        monitor.Tick += (_, _) => { if (!busy && wasRunning && !server.IsRunning) { wasRunning = false; Display(false); ErrorText.Text = "서버가 종료되었습니다. 다시 켜거나 로그를 확인해 주세요."; } }; monitor.Start();
        Closing += OnClosing;
    }
    protected override void OnSourceInitialized(EventArgs e)
    {
        base.OnSourceInitialized(e); var handle = new WindowInteropHelper(this).Handle;
        int backdrop = 2, corners = 2;
        DwmSetWindowAttribute(handle, 38, ref backdrop, sizeof(int)); DwmSetWindowAttribute(handle, 33, ref corners, sizeof(int));
    }
    [DllImport("dwmapi.dll")] static extern int DwmSetWindowAttribute(IntPtr handle, int attribute, ref int value, int size);
    string Key => VisibleKeyBox.Visibility == Visibility.Visible ? VisibleKeyBox.Text : KeyBox.Password;
    void RevealKey(object sender, RoutedEventArgs e)
    {
        if (VisibleKeyBox.Visibility == Visibility.Visible) { KeyBox.Password = VisibleKeyBox.Text; VisibleKeyBox.Visibility = Visibility.Collapsed; KeyBox.Visibility = Visibility.Visible; }
        else { VisibleKeyBox.Text = KeyBox.Password; KeyBox.Visibility = Visibility.Collapsed; VisibleKeyBox.Visibility = Visibility.Visible; }
    }
    async void ToggleServer(object sender, RoutedEventArgs e)
    {
        if (busy) return; busy = true; ErrorText.Text = ""; ToggleButton.IsEnabled = false; SettingsPanel.IsEnabled = false;
        try
        {
            if (server.IsRunning) { Status.Text = "서버 끄는 중"; StatusDetail.Text = "진행 중인 작업을 종료하고 있어요."; await server.StopAsync(); wasRunning = false; Display(false); }
            else
            {
                if (!int.TryParse(PortBox.Text, NumberStyles.None, CultureInfo.InvariantCulture, out var port) || port is < 1 or > 65535) throw new ArgumentException("Port must be a number between 1 and 65535.");
                var key = Key.Trim(); if (key.Length == 0 || key.Contains('\n') || key.Contains('\r')) throw new ArgumentException("Enter an API key.");
                settings.Save(port, key); Status.Text = "서버 켜는 중"; StatusDetail.Text = "실행 환경을 불러오고 있어요."; StatusDot.Fill = new SolidColorBrush(Color.FromRgb(0, 103, 192));
                await server.StartAsync(port, key); wasRunning = true; Display(true);
            }
        }
        catch (Exception error) { server.RecordError(error); Display(false); ErrorText.Text = error.Message; }
        finally { busy = false; ToggleButton.IsEnabled = true; SettingsPanel.IsEnabled = !server.IsRunning; }
    }
    void Display(bool running)
    {
        Status.Text = running ? "서버 실행 중" : "서버 꺼짐"; StatusDetail.Text = running ? server.Address : "설정을 입력하고 서버를 켜세요.";
        StatusDot.Fill = new SolidColorBrush(running ? Color.FromRgb(15, 123, 15) : Color.FromRgb(152, 152, 152));
        ToggleButton.Content = new System.Windows.Controls.TextBlock { Text = running ? "서버 끄기" : "서버 켜기", Foreground = Brushes.White }; OpenButton.IsEnabled = running; SettingsPanel.IsEnabled = !running;
    }
    void OpenWebsite(object sender, RoutedEventArgs e) { if (server.IsRunning) Process.Start(new ProcessStartInfo(server.Address) { UseShellExecute = true }); }
    async void OnClosing(object? sender, CancelEventArgs e)
    {
        if (allowClose) return;
        e.Cancel = true; IsEnabled = false;
        try { await server.StopAsync(); } finally { monitor.Stop(); server.Dispose(); allowClose = true; Close(); }
    }
    public void RenderPreview(string path)
    {
        Root.Margin = new Thickness(0); Root.Width = 424; Root.Height = 500; Root.Measure(new Size(424, 500)); Root.Arrange(new Rect(0, 0, 424, 500)); Root.UpdateLayout();
        var image = new RenderTargetBitmap(484, 560, 96, 96, PixelFormats.Pbgra32);
        var drawing = new DrawingVisual(); using (var dc = drawing.RenderOpen()) { dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(243, 243, 243)), null, new Rect(0, 0, 484, 560)); dc.PushTransform(new TranslateTransform(30, 26)); dc.DrawRectangle(new VisualBrush(Root), null, new Rect(0, 0, 424, 500)); }
        image.Render(drawing); var png = new PngBitmapEncoder(); png.Frames.Add(BitmapFrame.Create(image)); using var file = File.Create(path); png.Save(file);
    }
}
