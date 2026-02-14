"""Compare native vs ONNX at every pipeline stage using same codes."""
import numpy as np
import onnx, onnxruntime as ort
import os

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")
ONNX_MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\qwen3-tts-0.6b\tokenizer12hz_decode.onnx"

# Load codes dumped by native pipeline
codes = np.fromfile("voc_codes.raw", dtype=np.int64)
T = len(codes) // 16
codes = codes.reshape(1, T, 16)
print(f"Codes: [1, {T}, 16]")
print(f"  codes[0,:5] = {codes[0, 0, :5]}")

# Load ONNX model and add intermediate outputs
model = onnx.load(ONNX_MODEL, load_external_data=False)
model2 = onnx.ModelProto()
model2.CopyFrom(model)

targets_float = [
    "/decoder/pre_conv/conv/Conv_output_0",
    "/decoder/Transpose_19_output_0",
    "/decoder/upsample.0.0/conv/ConvTranspose_output_0",  # upsample0 transconv before convnext
    "/decoder/upsample.0.1/Add_output_0",                 # upsample0 after convnext
    "/decoder/upsample.1.0/conv/ConvTranspose_output_0",  # upsample1 transconv before convnext
    "/decoder/upsample.1.1/Add_output_0",                 # upsample1 after convnext
]

for t in targets_float:
    vi = onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
    model2.graph.output.append(vi)

print("Running ONNX model...")
sess = ort.InferenceSession(model2.SerializeToString())
results = sess.run(None, {"audio_codes": codes})
output_names = [o.name for o in sess.get_outputs()]

onnx_vals = {}
for i, (name, val) in enumerate(zip(output_names, results)):
    onnx_vals[name] = np.array(val)
    if name in targets_float or name in ("audio_values", "lengths"):
        arr = np.array(val)
        print(f"  {name}: shape={arr.shape}")

# Get valid length
lengths = int(onnx_vals["lengths"][0])
print(f"\nONNX valid length: {lengths} samples (= {lengths/1920} * 1920)")

# Compare at each stage
def compare(label, native_file, onnx_name, channels, onnx_idx=0):
    """Compare native dump with ONNX intermediate output."""
    if not os.path.exists(native_file):
        print(f"\n{label}: MISSING native file {native_file}")
        return
    native = np.fromfile(native_file, dtype=np.float32)
    T_native = len(native) // channels
    native = native.reshape(channels, T_native)

    onnx_arr = onnx_vals[onnx_name]
    if onnx_arr.ndim == 3:
        onnx_arr = onnx_arr[onnx_idx]  # [C, T_onnx] or similar
    T_onnx = onnx_arr.shape[-1] if onnx_arr.shape[0] == channels else onnx_arr.shape[0]

    # The ONNX output might have different shape ordering
    if onnx_arr.shape[0] != channels:
        # Try transposing or selecting first T_native
        if len(onnx_arr.shape) == 2 and onnx_arr.shape[1] == channels:
            onnx_arr = onnx_arr.T  # Was [T, C], now [C, T]
            T_onnx = onnx_arr.shape[1]

    # Compare valid positions
    cmp_T = min(T_native, T_onnx)
    native_cmp = native[:, :cmp_T]
    onnx_cmp = onnx_arr[:, :cmp_T]

    diff = np.abs(native_cmp - onnx_cmp)
    corr = np.corrcoef(native_cmp.flatten(), onnx_cmp.flatten())[0, 1]
    print(f"\n{label}: native [{channels}, {T_native}] vs ONNX [{onnx_arr.shape[0]}, {T_onnx if len(onnx_arr.shape) > 1 else 'N/A'}]")
    print(f"  max_diff={diff.max():.8f}, mean_diff={diff.mean():.8f}, corr={corr:.8f}")
    print(f"  native[0,:5]: {native[0,:5]}")
    print(f"  ONNX  [0,:5]: {onnx_cmp[0,:5]}")
    # Per-timestep
    for t in range(min(5, cmp_T)):
        dt = diff[:, t].max()
        print(f"  t={t}: max_diff={dt:.8f}")

compare("Pre-conv", "voc_xfmr_out.raw", "/decoder/Transpose_19_output_0", 1024)

# Wait - pre_conv comparison
compare("Pre-conv (raw)", "voc_xfmr_out.raw", "/decoder/pre_conv/conv/Conv_output_0", 1024)

# Transformer output
compare("Transformer", "voc_xfmr_out.raw", "/decoder/Transpose_19_output_0", 1024)

# Upsample 0 (after convnext)
compare("Upsample 0 (convnext)", "voc_upsample0_out.raw", "/decoder/upsample.0.1/Add_output_0", 1024)

# Upsample 1 (after convnext)
compare("Upsample 1 (convnext)", "voc_upsample1_out.raw", "/decoder/upsample.1.1/Add_output_0", 1024)

# Audio
print("\n=== Final Audio ===")
native_audio = np.fromfile("native_audio.raw", dtype=np.float32)
onnx_audio = np.fromfile("onnx_audio.raw", dtype=np.float32)
n = min(len(native_audio), len(onnx_audio))
corr = np.corrcoef(native_audio[:n], onnx_audio[:n])[0, 1]
snr_num = np.mean(onnx_audio[:n]**2)
snr_den = np.mean((native_audio[:n] - onnx_audio[:n])**2)
snr = 10 * np.log10(snr_num / snr_den) if snr_den > 0 else float('inf')
print(f"Audio: {n} samples, corr={corr:.6f}, SNR={snr:.1f} dB")

# Also extract and compare upsample transconv outputs (BEFORE convnext)
print("\n=== Upsample transconv (before convnext) ===")
for s, name in [(0, "/decoder/upsample.0.0/conv/ConvTranspose_output_0"),
                (1, "/decoder/upsample.1.0/conv/ConvTranspose_output_0")]:
    arr = onnx_vals[name]
    print(f"  Upsample {s} transconv: shape={arr.shape}")
    print(f"    [0,0,:5]: {arr[0,0,:5]}")
