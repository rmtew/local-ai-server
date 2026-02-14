"""Compare native vs ONNX vocoder audio output."""
import numpy as np
import os

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")

native = np.fromfile("native_audio.raw", dtype=np.float32)
onnx = np.fromfile("onnx_audio.raw", dtype=np.float32)

print(f"Native: {len(native)} samples ({len(native)/24000:.3f}s)")
print(f"ONNX:   {len(onnx)} samples ({len(onnx)/24000:.3f}s)")

n = min(len(native), len(onnx))
native_v = native[:n]
onnx_v = onnx[:n]

diff = np.abs(native_v - onnx_v)
corr = np.corrcoef(native_v, onnx_v)[0, 1]

sig_power = np.mean(onnx_v**2)
noise_power = np.mean((native_v - onnx_v)**2)
snr = 10 * np.log10(sig_power / noise_power) if noise_power > 0 else float('inf')

print(f"\nComparison ({n} samples):")
print(f"  Correlation: {corr:.6f}")
print(f"  SNR: {snr:.1f} dB")
print(f"  Max diff: {diff.max():.6f}")
print(f"  Mean diff: {diff.mean():.6f}")
print(f"  Native range: [{native_v.min():.4f}, {native_v.max():.4f}]")
print(f"  ONNX range:   [{onnx_v.min():.4f}, {onnx_v.max():.4f}]")

# Per-segment comparison
seg_size = 4800  # 200ms segments
n_segs = n // seg_size
print(f"\nPer-segment correlation ({seg_size} samples = 200ms):")
for i in range(min(n_segs, 10)):
    s = i * seg_size
    e = s + seg_size
    seg_corr = np.corrcoef(native_v[s:e], onnx_v[s:e])[0, 1]
    seg_diff = np.abs(native_v[s:e] - onnx_v[s:e]).max()
    print(f"  Seg {i}: corr={seg_corr:.6f}, max_diff={seg_diff:.6f}")
