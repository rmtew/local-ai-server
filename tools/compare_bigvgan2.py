"""Compare BigVGAN block 0 step by step."""
import numpy as np
import onnx, onnxruntime as ort
import os

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")
ONNX_MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\qwen3-tts-12hz-0.6b-base\tokenizer12hz_decode.onnx"

codes = np.fromfile("voc_codes.raw", dtype=np.int64).reshape(1, -1, 16)
T = codes.shape[1]
print(f"Codes: [1, {T}, 16]")

model = onnx.load(ONNX_MODEL, load_external_data=False)
model2 = onnx.ModelProto()
model2.CopyFrom(model)

targets = [
    # Init conv output
    "/decoder/decoder.0/conv/Conv_output_0",
    # Block 0: SnakeBeta output
    "/decoder/decoder.1/block.0/Add_1_output_0",
    # Block 0: ConvTranspose raw output
    "/decoder/decoder.1/block.1/conv/ConvTranspose_output_0",
    # Block 0: Slice output (trimmed ConvTranspose)
    "/decoder/decoder.1/block.1/Slice_output_0",
    # Block 0: ResUnit 0 SnakeBeta 1 output (after first snake in ResUnit)
    "/decoder/decoder.1/block.2/act1/Add_1_output_0",
    # Block 0: ResUnit 0 conv1 output
    "/decoder/decoder.1/block.2/conv1/conv/Conv_output_0",
    # Block 0: ResUnit 0 residual add
    "/decoder/decoder.1/block.2/Add_output_0",
    # Block 0 final output (after all ResUnits)
    "/decoder/decoder.1/block.4/Add_output_0",
]

for t in targets:
    vi = onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
    model2.graph.output.append(vi)

# Also check Slice node parameters for the ConvTranspose trim
print("\n=== Slice node in block 0 ===")
for node in model.graph.node:
    for out in node.output:
        if "/decoder/decoder.1/block.1/Slice" in out and node.op_type == "Slice":
            print(f"  Slice: inputs={list(node.input)}, output={out}")
            # Slice inputs: data, starts, ends, axes, steps
            for inp in node.input:
                # Check if any of these are constants
                for init in model.graph.initializer:
                    if init.name == inp:
                        arr = onnx.numpy_helper.to_array(init)
                        print(f"    {inp} = {arr}")
                # Also check Constant nodes
                for cnode in model.graph.node:
                    if cnode.op_type == "Constant":
                        for cout in cnode.output:
                            if cout == inp:
                                for attr in cnode.attribute:
                                    if attr.name == "value":
                                        arr = onnx.numpy_helper.to_array(attr.t)
                                        print(f"    {inp} = {arr}")

print("\nRunning ONNX model...")
sess = ort.InferenceSession(model2.SerializeToString())
results = sess.run(None, {"audio_codes": codes})
output_names = [o.name for o in sess.get_outputs()]

onnx_vals = {}
for i, (name, val) in enumerate(zip(output_names, results)):
    onnx_vals[name] = np.array(val)

for name in targets:
    arr = onnx_vals.get(name)
    if arr is not None:
        print(f"\n{name}: shape={arr.shape}")
        a = arr[0] if arr.ndim == 3 else arr
        T_valid = T  # 22 original timesteps
        # Show first values and native-equivalent range
        print(f"  [0,:5]: {a[0,:5]}")

# Now compare step by step with native
print("\n\n=== Step-by-step comparison ===")

# 1. Init conv: should match (already verified)
native_init = np.fromfile("voc_bigvgan_init.raw", dtype=np.float32).reshape(1536, -1)
onnx_init = onnx_vals["/decoder/decoder.0/conv/Conv_output_0"][0]
T_n = native_init.shape[1]  # 88
diff = np.abs(native_init - onnx_init[:, :T_n])
print(f"1. Init conv: max_diff={diff.max():.8f}")

# 2. SnakeBeta: compare native SnakeBeta(init_conv) with ONNX
# Native applies SnakeBeta to init_conv in-place before ConvTranspose
# ONNX SnakeBeta output is Add_1_output_0
onnx_snake = onnx_vals["/decoder/decoder.1/block.0/Add_1_output_0"][0]
print(f"2. SnakeBeta: ONNX shape={onnx_snake.shape}")
print(f"  ONNX [0,:5]: {onnx_snake[0,:5]}")

# We don't have native SnakeBeta dump, but we can compute it
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

alpha = load_st(SAFETENSORS, "decoder.decoder.1.block.0.alpha")  # [1, 1536, 1]
beta = load_st(SAFETENSORS, "decoder.decoder.1.block.0.beta")
exp_alpha = np.exp(alpha.flatten())
inv_exp_beta = 1.0 / np.exp(beta.flatten())

# Apply SnakeBeta to native init conv output
native_snake = native_init.copy()
for ch in range(1536):
    ea = exp_alpha[ch]
    ieb = inv_exp_beta[ch]
    for t in range(T_n):
        s = np.sin(ea * native_snake[ch, t])
        native_snake[ch, t] += ieb * s * s

diff = np.abs(native_snake - onnx_snake[:, :T_n])
print(f"  Py SnakeBeta vs ONNX: max_diff={diff.max():.8f}")
print(f"  Py [0,:5]: {native_snake[0,:5]}")

# 3. ConvTranspose
onnx_tconv_raw = onnx_vals["/decoder/decoder.1/block.1/conv/ConvTranspose_output_0"][0]
onnx_tconv_sliced = onnx_vals["/decoder/decoder.1/block.1/Slice_output_0"][0]
print(f"\n3. ConvTranspose:")
print(f"  Raw ONNX: {onnx_tconv_raw.shape}")
print(f"  Sliced ONNX: {onnx_tconv_sliced.shape}")
print(f"  Trimmed amount: {onnx_tconv_raw.shape[1] - onnx_tconv_sliced.shape[1]} from right/left")
# Check: is it trimmed from left or right?
# Compare first few values
print(f"  Raw  [0,:5]: {onnx_tconv_raw[0,:5]}")
print(f"  Slice[0,:5]: {onnx_tconv_sliced[0,:5]}")
# Check if slice starts from position 0 or later
if np.allclose(onnx_tconv_raw[0,:5], onnx_tconv_sliced[0,:5]):
    print(f"  -> Slice starts from position 0 (trims from right)")
else:
    # Find the offset
    for offset in range(onnx_tconv_raw.shape[1]):
        if np.allclose(onnx_tconv_raw[0,offset:offset+5], onnx_tconv_sliced[0,:5], atol=1e-7):
            print(f"  -> Slice starts from position {offset} (trims {offset} from left)")
            break

# Native ConvTranspose output: we don't have it dumped separately, but block 0 output includes ResUnits
# Let me compute native ConvTranspose from native_snake input
tconv_weight = load_st(SAFETENSORS, "decoder.decoder.1.block.1.conv.weight")  # [in_ch, out_ch, kernel]
tconv_bias = load_st(SAFETENSORS, "decoder.decoder.1.block.1.conv.bias")
print(f"  ConvTranspose weight: {tconv_weight.shape}")

# Compute native ConvTranspose1d
in_ch, out_ch, kernel = tconv_weight.shape  # [1536, 768, 16]
stride = 8
T_out = T_n * stride  # 704

native_tconv = np.zeros((out_ch, T_out), dtype=np.float32)
for ic in range(in_ch):
    for oc in range(out_ch):
        w = tconv_weight[ic, oc]
        x = native_snake[ic]
        for t in range(T_n):
            for ki in range(kernel):
                oi = t * stride + ki
                if oi < T_out:
                    native_tconv[oc, oi] += x[t] * w[ki]
for oc in range(out_ch):
    native_tconv[oc] += tconv_bias[oc]

# Compare native ConvTranspose with ONNX sliced
onnx_tconv_valid = onnx_tconv_sliced[:, :T_out]
diff = np.abs(native_tconv - onnx_tconv_valid)
print(f"  Native ConvTranspose vs ONNX sliced (first {T_out}): max_diff={diff.max():.8f}")
print(f"  Native [0,:5]: {native_tconv[0,:5]}")
print(f"  ONNX   [0,:5]: {onnx_tconv_valid[0,:5]}")
corr = np.corrcoef(native_tconv.flatten(), onnx_tconv_valid.flatten())[0, 1]
print(f"  Correlation: {corr:.8f}")
