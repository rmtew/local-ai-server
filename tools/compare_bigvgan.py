"""Compare BigVGAN stages between native and ONNX."""
import numpy as np
import onnx, onnxruntime as ort
import os

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")
ONNX_MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\qwen3-tts-0.6b\tokenizer12hz_decode.onnx"

# Load codes
codes = np.fromfile("voc_codes.raw", dtype=np.int64)
T = len(codes) // 16
codes = codes.reshape(1, T, 16)
print(f"Codes: [1, {T}, 16]")

# Load ONNX model with BigVGAN intermediate outputs
model = onnx.load(ONNX_MODEL, load_external_data=False)
model2 = onnx.ModelProto()
model2.CopyFrom(model)

# BigVGAN targets
targets = [
    "/decoder/decoder.0/conv/Conv_output_0",                         # init conv
    "/decoder/decoder.1/block.1/conv/ConvTranspose_output_0",        # block0 transconv
    "/decoder/decoder.1/block.4/conv2/conv/Conv_output_0",           # block0 last resunit conv2
    "/decoder/decoder.2/block.1/conv/ConvTranspose_output_0",        # block1 transconv
    "/decoder/decoder.3/block.1/conv/ConvTranspose_output_0",        # block2 transconv
    "/decoder/decoder.4/block.1/conv/ConvTranspose_output_0",        # block3 transconv
    "/decoder/decoder.6/conv/Conv_output_0",                         # final conv
]

# Also find the SnakeBeta outputs and residual adds
# Find Add nodes that are the residual connections in block 1
for node in model.graph.node:
    if node.op_type == "Add":
        for out in node.output:
            if "/decoder/decoder.1/" in out and "block" in out:
                targets.append(out)
                break

for t in targets:
    vi = onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
    model2.graph.output.append(vi)

print("Running ONNX model...")
sess = ort.InferenceSession(model2.SerializeToString())
results = sess.run(None, {"audio_codes": codes})
output_names = [o.name for o in sess.get_outputs()]

onnx_vals = {}
for i, (name, val) in enumerate(zip(output_names, results)):
    onnx_vals[name] = np.array(val)

def compare(label, native_data, onnx_name, channels):
    if onnx_name not in onnx_vals:
        print(f"\n{label}: MISSING ONNX output '{onnx_name}'")
        return
    onnx_arr = onnx_vals[onnx_name]
    if onnx_arr.ndim == 3:
        onnx_arr = onnx_arr[0]

    T_native = native_data.shape[1]
    T_onnx = onnx_arr.shape[1]
    cmp_T = min(T_native, T_onnx)

    native_cmp = native_data[:, :cmp_T]
    onnx_cmp = onnx_arr[:, :cmp_T]

    diff = np.abs(native_cmp - onnx_cmp)
    corr = np.corrcoef(native_cmp.flatten(), onnx_cmp.flatten())[0, 1]
    print(f"\n{label}: native [{channels}, {T_native}] vs ONNX [{onnx_arr.shape[0]}, {T_onnx}]")
    print(f"  max_diff={diff.max():.8f}, mean_diff={diff.mean():.8f}, corr={corr:.8f}")
    print(f"  native[0,:5]: {native_data[0,:min(5,T_native)]}")
    print(f"  ONNX  [0,:5]: {onnx_cmp[0,:min(5,cmp_T)]}")

# BigVGAN init conv
native_init = np.fromfile("voc_bigvgan_init.raw", dtype=np.float32)
ch = 1536
T_init = len(native_init) // ch
native_init = native_init.reshape(ch, T_init)
compare("BigVGAN init conv", native_init, "/decoder/decoder.0/conv/Conv_output_0", ch)

# BigVGAN block 0 transconv output
onnx_b0tc = onnx_vals.get("/decoder/decoder.1/block.1/conv/ConvTranspose_output_0")
if onnx_b0tc is not None:
    print(f"\nONNX block0 transconv shape: {onnx_b0tc.shape}")
    print(f"  [0,0,:5]: {onnx_b0tc[0,0,:5]}")

# BigVGAN block 0 output (from native dump)
native_b0 = np.fromfile("voc_bigvgan_blk0.raw", dtype=np.float32)
ch0 = 768
T_b0 = len(native_b0) // ch0
native_b0 = native_b0.reshape(ch0, T_b0)
print(f"\nNative block0 output: [{ch0}, {T_b0}]")
print(f"  [0,:5]: {native_b0[0,:5]}")

# Find block 0 output in ONNX (after last residual add in decoder.1)
# The last ResUnit is block.4, and its output has a residual add
# Let's find what feeds into decoder.2/block.0 (which is the SnakeBeta of block 1)
for node in model.graph.node:
    for out in node.output:
        if "/decoder/decoder.2/block.0/" in out and "Add_1" in out:
            # This is the SnakeBeta output, its input is the block 0 output
            for inp in node.input:
                print(f"\nBlock 1 SnakeBeta input ({out}): inputs={list(node.input)}")
            break

# Actually, let me find the Add nodes that are residual adds in decoder.1
print("\n=== Residual Add nodes in decoder.1 ===")
for node in model.graph.node:
    if node.op_type == "Add":
        for out in node.output:
            if "/decoder/decoder.1/" in out:
                print(f"  Add: inputs={list(node.input)}, output={out}")

# Let me also look at what decoder.2/block.0/Add_1 feeds on
print("\n=== What feeds into block 1 (decoder.2) ===")
for node in model.graph.node:
    for out in node.output:
        if "/decoder/decoder.2/block.0/Add_1" in out:
            print(f"  {node.op_type}: inputs={list(node.input)}, output={out}")

# Check decoder.1/block.4 Add (residual) output
for node in model.graph.node:
    for out in node.output:
        if "/decoder/decoder.1/block.4/" in out and "Add" in out and node.op_type == "Add":
            print(f"\n  Block0 ResUnit2 Add: inputs={list(node.input)}, output={out}")
            # Try to compare this with native block 0 output
            if out in onnx_vals:
                compare("Block0 ResUnit2 Add", native_b0, out, ch0)
