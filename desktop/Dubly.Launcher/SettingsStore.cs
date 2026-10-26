using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Dubly.Launcher;

public sealed record Settings(int Port = 20806, string EncryptedApiKey = "");

public sealed class SettingsStore
{
    public string DirectoryPath { get; }
    public SettingsStore(string? directory = null) => DirectoryPath = directory ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Dubly");
    public (int Port, string Key) Load()
    {
        try
        {
            var value = JsonSerializer.Deserialize<Settings>(File.ReadAllText(Path.Combine(DirectoryPath, "settings.json"))) ?? new();
            return (value.Port is >= 1 and <= 65535 ? value.Port : 20806,
                value.EncryptedApiKey.Length == 0 ? "" : Encoding.UTF8.GetString(Protect(Convert.FromBase64String(value.EncryptedApiKey), false)));
        }
        catch (Exception error) when (error is IOException or JsonException or FormatException or CryptographicException) { return (20806, ""); }
    }
    public void Save(int port, string key)
    {
        Directory.CreateDirectory(DirectoryPath);
        var value = new Settings(port, Convert.ToBase64String(Protect(Encoding.UTF8.GetBytes(key), true)));
        var path = Path.Combine(DirectoryPath, "settings.json");
        var temp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        File.WriteAllText(temp, JsonSerializer.Serialize(value), new UTF8Encoding(false));
        File.Move(temp, path, true);
    }
    [StructLayout(LayoutKind.Sequential)] struct Blob { public int Length; public IntPtr Data; }
    [DllImport("crypt32.dll", SetLastError = true, CharSet = CharSet.Unicode)] static extern bool CryptProtectData(ref Blob input, string? description, IntPtr entropy, IntPtr reserved, IntPtr prompt, int flags, out Blob output);
    [DllImport("crypt32.dll", SetLastError = true, CharSet = CharSet.Unicode)] static extern bool CryptUnprotectData(ref Blob input, IntPtr description, IntPtr entropy, IntPtr reserved, IntPtr prompt, int flags, out Blob output);
    [DllImport("kernel32.dll")] static extern IntPtr LocalFree(IntPtr memory);
    static byte[] Protect(byte[] data, bool encrypt)
    {
        var input = new Blob { Length = data.Length, Data = Marshal.AllocHGlobal(data.Length) };
        Blob output = default;
        try
        {
            Marshal.Copy(data, 0, input.Data, data.Length);
            var ok = encrypt ? CryptProtectData(ref input, "Dubly API key", IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 1, out output)
                : CryptUnprotectData(ref input, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 1, out output);
            if (!ok) throw new CryptographicException(Marshal.GetLastWin32Error());
            var result = new byte[output.Length]; Marshal.Copy(output.Data, result, 0, result.Length); return result;
        }
        finally { Marshal.FreeHGlobal(input.Data); if (output.Data != IntPtr.Zero) LocalFree(output.Data); }
    }
}
