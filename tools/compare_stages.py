"""Compare native vs ONNX at each pipeline stage using the dump files."""
import numpy as np
import onnx, onnxruntime as ort
import os

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")
ONNX_MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\qwen3-tts-0.6b\tokenizer12hz_decode.onnx"

# Load native dumps - determine T from transformer output
native_xfmr = np.fromfile("voc_xfmr_out.raw", dtype=np.float32)
T = len(native_xfmr) // 1024
native_xfmr = native_xfmr.reshape(1024, T)
print(f"Native T = {T}")

# We need the same codes that produced these dumps
# Read codes from the vocoder - they're printed by the pipeline in verbose mode
# For now, let's load the codes from the native pipeline's last run
# Actually, let's just load from the raw dumps and compare shapes

stages = [
    ("voc_xfmr_out.raw", 1024, "Transformer output"),
    ("voc_upsample0_out.raw", 1024, "Upsample 0 (after convnext)"),
    ("voc_upsample1_out.raw", 1024, "Upsample 1 (after convnext)"),
    ("voc_bigvgan_init.raw", 1536, "BigVGAN init conv"),
    ("voc_bigvgan_blk0.raw", 768, "BigVGAN block 0"),
    ("voc_bigvgan_blk1.raw", 384, "BigVGAN block 1"),
    ("voc_bigvgan_blk2.raw", 192, "BigVGAN block 2"),
    ("voc_bigvgan_blk3.raw", 96, "BigVGAN block 3"),
]

for fname, channels, desc in stages:
    if os.path.exists(fname):
        data = np.fromfile(fname, dtype=np.float32)
        T_stage = len(data) // channels
        data = data.reshape(channels, T_stage)
        print(f"\n{desc}: [{channels}, {T_stage}]")
        print(f"  range: [{data.min():.6f}, {data.max():.6f}]")
        print(f"  [0,:5]: {data[0,:5]}")
        if channels > 1:
            print(f"  [1,:5]: {data[1,:5]}")
    else:
        print(f"\n{desc}: MISSING ({fname})")

# Also compare native audio with ONNX audio
print("\n=== Audio comparison ===")
native_audio = np.fromfile("native_audio.raw", dtype=np.float32)
onnx_audio = np.fromfile("onnx_audio.raw", dtype=np.float32)
n = min(len(native_audio), len(onnx_audio))
print(f"Native: {len(native_audio)} samples, ONNX: {len(onnx_audio)} samples")
native_audio = native_audio[:n]
onnx_audio = onnx_audio[:n]
corr = np.corrcoef(native_audio, onnx_audio)[0, 1]
snr_num = np.mean(onnx_audio**2)
snr_den = np.mean((native_audio - onnx_audio)**2)
snr = 10 * np.log10(snr_num / snr_den) if snr_den > 0 else float('inf')
print(f"Correlation: {corr:.6f}, SNR: {snr:.1f} dB")

# Key: Compare native upsample0 output with ONNX
# The ONNX upsample.0.1/Add_output_0 is the ConvNeXt output after first upsample
# Let's also check the upsample0 transconv output BEFORE convnext
print("\n=== Checking transformer output alignment ===")
# The native xfmr output has T timesteps
# The ONNX xfmr output has 1024 timesteps (padded)
# After the epsilon fix, these should be closer for positions 0..T-1
# But we can't run ONNX with the same codes here (we don't know them)
# Let's check if the dump file values look reasonable
print(f"Native xfmr [0,:10]: {native_xfmr[0,:10]}")
print(f"Native xfmr [-1,:10]: {native_xfmr[-1,:10]}")
print(f"Native xfmr stats: mean={native_xfmr.mean():.6f}, std={native_xfmr.std():.6f}")

# Check if there's a structural issue - does the audio shape match?
expected_audio = T * 1920
print(f"\nExpected audio samples: {T} * 1920 = {expected_audio}")
print(f"Actual native audio samples: {len(np.fromfile('native_audio.raw', dtype=np.float32))}")
print(f"Actual ONNX audio samples: {len(np.fromfile('onnx_audio.raw', dtype=np.float32))}")
