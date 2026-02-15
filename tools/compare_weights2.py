"""Deep comparison: extract matching weights from ONNX and safetensors."""
import struct, json, numpy as np, os
os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")

SAFETENSORS = r"C:\Data\R\git\claude-repos\deps\models\tts\Qwen3-TTS-Tokenizer-12Hz\model.safetensors"
ONNX_MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\qwen3-tts-12hz-0.6b-base\tokenizer12hz_decode.onnx"

def load_st_tensor(path, name):
    with open(path, "rb") as f:
        hs = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hs))
        ds = 8 + hs
        info = hdr[name]
        b, e = info["data_offsets"]
        f.seek(ds + b)
        return np.frombuffer(f.read(e - b), dtype=np.float32).copy().reshape(info["shape"])

import onnx
model = onnx.load(ONNX_MODEL, load_external_data=False)

# Build ONNX weight dict
onnx_weights = {}
for init in model.graph.initializer:
    onnx_weights[init.name] = onnx.numpy_helper.to_array(init)

# Load safetensors header
with open(SAFETENSORS, "rb") as f:
    hs = struct.unpack("<Q", f.read(8))[0]
    st_hdr = json.loads(f.read(hs))

# Compare all matching names
print("=== Matching weight comparison ===")
matches = 0
mismatches = 0
for name in sorted(onnx_weights.keys()):
    if name in st_hdr and name != "__metadata__":
        st = load_st_tensor(SAFETENSORS, name)
        ox = onnx_weights[name]
        if st.shape == ox.shape:
            diff = np.abs(st.flatten() - ox.flatten()).max()
            if diff > 1e-6:
                print(f"  DIFFER: {name}: {st.shape}, max_diff={diff:.8f}")
                mismatches += 1
            else:
                matches += 1
        else:
            print(f"  SHAPE: {name}: st={st.shape} onnx={ox.shape}")
            mismatches += 1

print(f"\n  Matched: {matches}, Mismatched: {mismatches}")

# Show ONNX-only weights (not in safetensors)
print(f"\n=== ONNX-only weights (not in safetensors) ===")
for name in sorted(onnx_weights.keys()):
    if name not in st_hdr:
        w = onnx_weights[name]
        print(f"  {name}: shape={list(w.shape)}, range=[{w.min():.4f}, {w.max():.4f}]")

# Now try to match ONNX-only weights to safetensors by shape and content
print(f"\n=== Matching ONNX-only to safetensors by shape ===")
st_decoder_names = [n for n in st_hdr if n.startswith("decoder.") and n != "__metadata__"]
for onnx_name in sorted(onnx_weights.keys()):
    if onnx_name in st_hdr:
        continue
    ow = onnx_weights[onnx_name]
    # Find safetensors weights with same shape
    candidates = []
    for sn in st_decoder_names:
        si = st_hdr[sn]
        if si["shape"] == list(ow.shape):
            candidates.append(sn)
    if candidates and len(candidates) <= 20:
        # Try comparing each
        for sn in candidates:
            st = load_st_tensor(SAFETENSORS, sn)
            diff = np.abs(st.flatten() - ow.flatten()).max()
            if diff < 1e-4:
                print(f"  {onnx_name} = {sn} (max_diff={diff:.8f})")
                break
            # Also try transposed
            if len(ow.shape) == 2:
                diff_t = np.abs(st.flatten() - ow.T.flatten()).max()
                if diff_t < 1e-4:
                    print(f"  {onnx_name} = {sn}.T (TRANSPOSED, max_diff={diff_t:.8f})")
                    break
