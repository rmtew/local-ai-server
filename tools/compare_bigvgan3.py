"""Compare BigVGAN block 0: extract ONNX Slice output and compare with native."""
import numpy as np
import onnx, onnxruntime as ort
import os

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")
ONNX_MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\qwen3-tts-0.6b\tokenizer12hz_decode.onnx"

codes = np.fromfile("voc_codes.raw", dtype=np.int64).reshape(1, -1, 16)
T = codes.shape[1]

model = onnx.load(ONNX_MODEL, load_external_data=False)

# First, find the Slice parameters
print("=== Slice node in block 0 ===")
for node in model.graph.node:
    if node.op_type == "Slice":
        for out in node.output:
            if "/decoder/decoder.1/block.1/Slice" in out:
                print(f"Slice: inputs={list(node.input)}, output={out}")

# Check what the Slice constant inputs are
# The Slice node uses: data, starts, ends, axes, steps
# Look for Constant nodes that feed into the Slice
slice_inputs = []
for node in model.graph.node:
    if node.op_type == "Slice":
        for out in node.output:
            if "/decoder/decoder.1/block.1/Slice" in out:
                slice_inputs = list(node.input)

print(f"Slice inputs: {slice_inputs}")
for inp_name in slice_inputs[1:]:  # skip data input
    for node in model.graph.node:
        if node.op_type == "Constant":
            for out in node.output:
                if out == inp_name:
                    for attr in node.attribute:
                        if attr.name == "value":
                            arr = onnx.numpy_helper.to_array(attr.t)
                            print(f"  {inp_name} = {arr}")

# Run ONNX and extract key outputs
model2 = onnx.ModelProto()
model2.CopyFrom(model)

targets = [
    "/decoder/decoder.0/conv/Conv_output_0",
    "/decoder/decoder.1/block.0/Add_1_output_0",          # SnakeBeta output
    "/decoder/decoder.1/block.1/conv/ConvTranspose_output_0",  # raw transconv
    "/decoder/decoder.1/block.1/Slice_output_0",           # trimmed transconv
    "/decoder/decoder.1/block.2/Add_output_0",             # after ResUnit 0
    "/decoder/decoder.1/block.4/Add_output_0",             # after all ResUnits
]
for t in targets:
    vi = onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
    model2.graph.output.append(vi)

print("\nRunning ONNX...")
sess = ort.InferenceSession(model2.SerializeToString())
results = sess.run(None, {"audio_codes": codes})
output_names = [o.name for o in sess.get_outputs()]
onnx_vals = {}
for i, (name, val) in enumerate(zip(output_names, results)):
    onnx_vals[name] = np.array(val)

# Print shapes
for t in targets:
    arr = onnx_vals[t]
    print(f"  {t}: {arr.shape}")

# Key comparison: look at the ConvTranspose trim
raw = onnx_vals["/decoder/decoder.1/block.1/conv/ConvTranspose_output_0"][0]
sliced = onnx_vals["/decoder/decoder.1/block.1/Slice_output_0"][0]
print(f"\nConvTranspose: raw={raw.shape}, sliced={sliced.shape}")
print(f"  Trim amount: {raw.shape[1] - sliced.shape[1]}")

# Find where sliced starts in raw
for offset in range(min(20, raw.shape[1])):
    if np.allclose(raw[:, offset:offset+5], sliced[:, :5], atol=1e-7):
        print(f"  Slice offset from start: {offset}")
        break
else:
    print("  Could not find offset!")

# For our native causal ConvTranspose: output = T*stride, trimming (kernel-stride) from RIGHT
# kernel=16, stride=8 -> trim 8 from right
# Full output: (88-1)*8 + 16 = 712. Trim 8 -> 704.
# ONNX full output: (4096-1)*8 + 16 = 32783. After slice...
# The sliced shape tells us: the ONNX trims to get sliced.shape[1]
# If ONNX trims from the LEFT, that would explain the divergence!

# For the valid 88 timesteps -> 704 output timesteps,
# check if the ONNX sliced output at positions 0-703 matches what our native produces

# Load native block 0 output
native_b0 = np.fromfile("voc_bigvgan_blk0.raw", dtype=np.float32)
ch = 768
T_b0 = len(native_b0) // ch
native_b0 = native_b0.reshape(ch, T_b0)

# Load native init conv output (input to SnakeBeta)
native_init = np.fromfile("voc_bigvgan_init.raw", dtype=np.float32).reshape(1536, -1)
T_init = native_init.shape[1]

# Compare init conv
onnx_init = onnx_vals["/decoder/decoder.0/conv/Conv_output_0"][0]
diff_init = np.abs(native_init - onnx_init[:, :T_init])
print(f"\nInit conv: max_diff={diff_init.max():.8f} (PASS)")

# Compare SnakeBeta
onnx_snake = onnx_vals["/decoder/decoder.1/block.0/Add_1_output_0"][0]
# We need to compute native SnakeBeta from init conv
import struct, json
SAFETENSORS = r"C:\Data\R\git\claude-repos\deps\models\tts\Qwen3-TTS-Tokenizer-12Hz\model.safetensors"
def load_st(path, name):
    with open(path, 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hs))
        ds = 8 + hs
        info = hdr[name]
        b, e = info['data_offsets']
        f.seek(ds + b)
        return np.frombuffer(f.read(e - b), dtype=np.float32).copy().reshape(info['shape'])

alpha = load_st(SAFETENSORS, "decoder.decoder.1.block.0.alpha").flatten()
beta = load_st(SAFETENSORS, "decoder.decoder.1.block.0.beta").flatten()
exp_alpha = np.exp(alpha)
inv_exp_beta = 1.0 / np.exp(beta)

# Vectorized SnakeBeta
native_snake = native_init.copy()
for ch in range(1536):
    ea = exp_alpha[ch]
    ieb = inv_exp_beta[ch]
    s = np.sin(ea * native_snake[ch])
    native_snake[ch] += ieb * s * s

diff_snake = np.abs(native_snake - onnx_snake[:, :T_init])
print(f"SnakeBeta: max_diff={diff_snake.max():.8f}")
print(f"  Py[0,:5]: {native_snake[0,:5]}")
print(f"  OX[0,:5]: {onnx_snake[0,:5]}")

# Now compute native ConvTranspose using numpy (vectorized)
tconv_weight = load_st(SAFETENSORS, "decoder.decoder.1.block.1.conv.weight")
tconv_bias = load_st(SAFETENSORS, "decoder.decoder.1.block.1.conv.bias")
in_ch, out_ch, kernel = tconv_weight.shape  # [1536, 768, 16]
stride = 8
T_out_native = T_init * stride  # 704

# Vectorized ConvTranspose: for each kernel position, matmul + scatter
native_tconv = np.zeros((out_ch, T_out_native), dtype=np.float32)
for ki in range(kernel):
    # W_ki: [in_ch, out_ch] -> contribution from all channels at kernel pos ki
    W_ki = tconv_weight[:, :, ki]  # [in_ch, out_ch]
    # contrib: [out_ch, T_init] = W_ki.T @ x
    contrib = W_ki.T @ native_snake  # [out_ch, T_init]
    # Scatter to output positions: t*stride + ki for each input position t
    for t in range(T_init):
        oi = t * stride + ki
        if oi < T_out_native:
            native_tconv[:, oi] += contrib[:, t]
native_tconv += tconv_bias[:, None]

# Compare with ONNX sliced ConvTranspose
onnx_sliced = onnx_vals["/decoder/decoder.1/block.1/Slice_output_0"][0]
onnx_tconv_valid = onnx_sliced[:, :T_out_native]
diff_tconv = np.abs(native_tconv - onnx_tconv_valid)
corr_tconv = np.corrcoef(native_tconv.flatten(), onnx_tconv_valid.flatten())[0, 1]
print(f"\nConvTranspose (native vs ONNX sliced):")
print(f"  max_diff={diff_tconv.max():.8f}, mean_diff={diff_tconv.mean():.8f}")
print(f"  corr={corr_tconv:.8f}")
print(f"  Py[0,:5]: {native_tconv[0,:5]}")
print(f"  OX[0,:5]: {onnx_tconv_valid[0,:5]}")
for t in range(min(10, T_out_native)):
    dt = diff_tconv[:, t].max()
    print(f"  t={t}: max_diff={dt:.8f}")
