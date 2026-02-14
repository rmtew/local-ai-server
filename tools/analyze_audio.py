"""Analyze native vs ONNX vocoder audio to find the nature of the difference."""
import numpy as np
import os

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")

native = np.fromfile("native_audio.raw", dtype=np.float32)
onnx = np.fromfile("onnx_audio.raw", dtype=np.float32)

print(f"Native: {len(native)} samples, range [{native.min():.4f}, {native.max():.4f}], RMS={np.sqrt(np.mean(native**2)):.6f}")
print(f"ONNX:   {len(onnx)} samples, range [{onnx.min():.4f}, {onnx.max():.4f}], RMS={np.sqrt(np.mean(onnx**2)):.6f}")

# Cross-correlation to check for time offset
from numpy.fft import fft, ifft
n = len(native)
xcorr = np.real(ifft(fft(native, 2*n) * np.conj(fft(onnx, 2*n))))[:n]
peak_offset = np.argmax(np.abs(xcorr))
print(f"\nCross-correlation peak at offset {peak_offset} (0 = aligned)")
print(f"  peak value: {xcorr[peak_offset]:.6f}")
print(f"  value at 0: {xcorr[0]:.6f}")

# Check if native is a scaled version of ONNX
# Linear regression: native = a * onnx + b
if len(native) > 10:
    A = np.vstack([onnx, np.ones(len(onnx))]).T
    result = np.linalg.lstsq(A, native, rcond=None)
    a, b = result[0]
    print(f"\nLinear fit: native = {a:.4f} * onnx + {b:.6f}")
    residual = native - (a * onnx + b)
    print(f"  Residual RMS: {np.sqrt(np.mean(residual**2)):.6f}")
    print(f"  R-squared: {1 - np.var(residual) / np.var(native):.6f}")

# Compare energy in different frequency bands
print("\nSpectral analysis:")
native_fft = np.abs(fft(native))[:n//2]
onnx_fft = np.abs(fft(onnx))[:n//2]
freqs = np.arange(n//2) * 24000 / n

bands = [(0, 100), (100, 500), (500, 2000), (2000, 6000), (6000, 12000)]
for lo, hi in bands:
    mask = (freqs >= lo) & (freqs < hi)
    native_e = np.sqrt(np.mean(native_fft[mask]**2))
    onnx_e = np.sqrt(np.mean(onnx_fft[mask]**2))
    ratio = native_e / (onnx_e + 1e-10)
    print(f"  {lo:5d}-{hi:5d} Hz: native={native_e:.4f}, onnx={onnx_e:.4f}, ratio={ratio:.2f}")

# Sample-level comparison at different positions
print("\nSample comparison at various positions:")
positions = [0, 100, 500, 1000, 2000, 4000, n//2, n-1000, n-100, n-1]
for p in positions:
    if p < n:
        print(f"  [{p:6d}] native={native[p]:10.6f}  onnx={onnx[p]:10.6f}  ratio={native[p]/(onnx[p]+1e-10):8.2f}")

# Check if it's the same signal with a phase/time shift
print("\nChecking time offsets (cross-corr at various lags):")
for lag in [0, 1, 2, 3, 4, 5, 8, 16, 100, 960, 1920]:
    if lag < n:
        corr = np.corrcoef(native[lag:], onnx[:n-lag])[0,1] if n-lag > 1 else 0
        print(f"  lag={lag:5d}: correlation={corr:.4f}")
