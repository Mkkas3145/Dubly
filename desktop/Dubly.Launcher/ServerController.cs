using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Dubly.Launcher;

public sealed class ServerController : IDisposable
{
    readonly string bundleRoot, logDirectory;
    readonly HttpClient http = new(new HttpClientHandler { UseProxy = false }) { Timeout = TimeSpan.FromSeconds(2) };
    Process? process; IntPtr job; string token = "", key = "";
    public int Port { get; private set; }
    public bool IsRunning => process is { HasExited: false };
    public string Address => $"http://localhost:{Port}";
    public ServerController(string root, string logDirectory) { bundleRoot = Path.GetFullPath(root); this.logDirectory = logDirectory; }

    public async Task StartAsync(int port, string apiKey)
    {
        if (IsRunning) throw new InvalidOperationException("The server is already running.");
        if (port is < 1 or > 65535) throw new ArgumentException("Port must be a number between 1 and 65535.");
        if (string.IsNullOrWhiteSpace(apiKey) || apiKey.Contains('\r') || apiKey.Contains('\n')) throw new ArgumentException("Enter a valid API key.");
        var listener = new TcpListener(IPAddress.Any, port) { ExclusiveAddressUse = true };
        try { listener.Start(); } catch (SocketException) { throw new InvalidOperationException("Another application is using this port. Choose a different port."); } finally { listener.Stop(); }
        var python = Path.Combine(bundleRoot, "runtime", "python", "python.exe");
        var server = Path.Combine(bundleRoot, "server");
        if (!File.Exists(python) || !File.Exists(Path.Combine(server, "app.py"))) throw new InvalidOperationException("Server runtime files are missing. Reinstall Dubly.");
        Directory.CreateDirectory(logDirectory); Port = port; key = apiKey.Trim(); token = Convert.ToHexString(RandomNumberGenerator.GetBytes(32));
        var info = new ProcessStartInfo(python) { UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = server,
            RedirectStandardOutput = true, RedirectStandardError = true, StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8 };
        foreach (var arg in new[] { "-X", "utf8", "-u", Path.Combine(server, "app.py") }) info.ArgumentList.Add(arg);
        info.Environment["OPENROUTER_API_KEY"] = key;
        info.Environment["DUBLY_PORT"] = port.ToString(); info.Environment["DUBLY_LAUNCHER_TOKEN"] = token;
        info.Environment["PATH"] = Path.Combine(bundleRoot, "runtime", "bin") + ";" + (info.Environment["PATH"] ?? "");
        info.Environment["HF_HOME"] = Path.Combine(server, "work", "model-cache");
        info.Environment["PYTHONUTF8"] = "1";
        process = new Process { StartInfo = info };
        process.OutputDataReceived += (_, e) => Log(e.Data); process.ErrorDataReceived += (_, e) => Log(e.Data);
        try
        {
            if (!process.Start()) throw new InvalidOperationException("Failed to start the server.");
            AttachJob(process.Handle); process.BeginOutputReadLine(); process.BeginErrorReadLine();
            var started = Stopwatch.StartNew();
            while (started.Elapsed < TimeSpan.FromSeconds(90))
            {
                if (!IsRunning) throw new InvalidOperationException("The server failed to start. Check the log.");
                try
                {
                    using var request = Request(HttpMethod.Get, "/api/launcher/health");
                    using var response = await http.SendAsync(request);
                    if (response.IsSuccessStatusCode)
                    {
                        using var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
                        var value = body.RootElement;
                        if (value.TryGetProperty("service", out var service) && service.GetString() == "dubly" && value.TryGetProperty("managed", out var managed) && managed.GetBoolean()) return;
                        throw new InvalidOperationException("Another server is running on this port. Choose a different port.");
                    }
                }
                catch (Exception error) when (error is HttpRequestException or TaskCanceledException or JsonException) { }
                await Task.Delay(250);
            }
            throw new InvalidOperationException("Server startup timed out. Check the log.");
        }
        catch { await StopAsync(); throw; }
    }
    HttpRequestMessage Request(HttpMethod method, string route)
    {
        var request = new HttpRequestMessage(method, $"http://127.0.0.1:{Port}{route}");
        request.Headers.Authorization = new("Bearer", token); return request;
    }
    public async Task StopAsync()
    {
        var owned = process;
        if (owned is null) return;
        if (!owned.HasExited)
        {
            try { using var request = Request(HttpMethod.Post, "/api/launcher/shutdown"); using var response = await http.SendAsync(request); }
            catch (Exception error) when (error is HttpRequestException or TaskCanceledException) { }
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(15));
            try { await owned.WaitForExitAsync(timeout.Token); }
            catch (OperationCanceledException) { if (!owned.HasExited) owned.Kill(entireProcessTree: true); await owned.WaitForExitAsync(); }
        }
        owned.WaitForExit(); process = null; owned.Dispose(); CloseJob(); key = ""; token = "";
    }
    public void RecordError(Exception error) => Log(error.ToString());
    readonly object logLock = new();
    void Log(string? line)
    {
        if (line is null) return;
        if (key.Length > 0) line = line.Replace(key, "[REDACTED]", StringComparison.Ordinal);
        try { lock (logLock) File.AppendAllText(Path.Combine(logDirectory, "server.log"), DateTime.Now.ToString("s") + " " + line + Environment.NewLine, Encoding.UTF8); }
        catch (IOException) { }
    }
    void AttachJob(IntPtr processHandle)
    {
        job = CreateJobObject(IntPtr.Zero, null); if (job == IntPtr.Zero) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "Failed to configure the server process job.");
        var limits = new ExtendedLimits { Basic = new BasicLimits { LimitFlags = 0x2000 } };
        var length = Marshal.SizeOf<ExtendedLimits>(); var memory = Marshal.AllocHGlobal(length);
        try
        {
            Marshal.StructureToPtr(limits, memory, false);
            if (!SetInformationJobObject(job, 9, memory, (uint)length) || !AssignProcessToJobObject(job, processHandle)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "Failed to configure the server process job.");
        }
        finally { Marshal.FreeHGlobal(memory); }
    }
    void CloseJob() { if (job != IntPtr.Zero) { CloseHandle(job); job = IntPtr.Zero; } }
    public void Dispose() { if (IsRunning) process!.Kill(entireProcessTree: true); process?.Dispose(); CloseJob(); http.Dispose(); }
    [StructLayout(LayoutKind.Sequential)] struct BasicLimits { public long UserTime, JobTime; public uint LimitFlags; public UIntPtr MinWorkingSet, MaxWorkingSet; public uint ActiveProcesses; public UIntPtr Affinity; public uint Priority, Scheduling; }
    [StructLayout(LayoutKind.Sequential)] struct IoCounters { public ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes; }
    [StructLayout(LayoutKind.Sequential)] struct ExtendedLimits { public BasicLimits Basic; public IoCounters Io; public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory; }
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] static extern IntPtr CreateJobObject(IntPtr attributes, string? name);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool SetInformationJobObject(IntPtr job, int type, IntPtr info, uint length);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
}
