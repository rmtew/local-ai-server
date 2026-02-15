"""Match ONNX-only weights to safetensors by value comparison."""
import struct, json, numpy as np, os, onnx
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

with open(SAFETENSORS, "rb") as f:
    hs = struct.unpack("<Q", f.read(8))[0]
    st_hdr = json.loads(f.read(hs))

model = onnx.load(ONNX_MODEL, load_external_data=False)
onnx_weights = {}
for init in model.graph.initializer:
    onnx_weights[init.name] = onnx.numpy_helper.to_array(init)

# Get all decoder weights from safetensors (not already matched)
matched_names = set(n for n in onnx_weights if n in st_hdr)
unmatched_onnx = sorted(n for n in onnx_weights if n not in st_hdr)
st_decoder = sorted(n for n in st_hdr if n.startswith("decoder.") and n != "__metadata__" and n not in matched_names)

print(f"ONNX-only weights to match: {len(unmatched_onnx)}")
print(f"Unmatched safetensors decoder weights: {len(st_decoder)}")

# For each unmatched ONNX weight, find matching safetensors by value
for oname in unmatched_onnx:
    ow = onnx_weights[oname]
    ow_flat = ow.flatten()
    found = False
    for sname in st_decoder:
        si = st_hdr[sname]
        st_size = 1
        for s in si["shape"]:
            st_size *= s
        # Check same number of elements
        if st_size != ow_flat.size:
            continue
        st = load_st_tensor(SAFETENSORS, sname)
        st_flat = st.flatten()
        diff = np.abs(st_flat - ow_flat).max()
        if diff < 1e-6:
            print(f"  {oname} {list(ow.shape)} = {sname} {si['shape']} (exact)")
            found = True
            break
        # Try transposed
        if len(ow.shape) == 2 and len(si["shape"]) == 2:
            diff_t = np.abs(st_flat - ow.T.flatten()).max()
            if diff_t < 1e-6:
                print(f"  {oname} {list(ow.shape)} = {sname} {si['shape']} (TRANSPOSED)")
                found = True
                break
    if not found:
        print(f"  {oname} {list(ow.shape)}: NO MATCH (range [{ow.min():.4f}, {ow.max():.4f}])")

# Also inspect the ONNX graph structure briefly
print(f"\n=== ONNX graph node types ===")
node_types = {}
for node in model.graph.node:
    node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
for t, c in sorted(node_types.items()):
    print(f"  {t}: {c}")

# Check inputs/outputs
print(f"\n=== ONNX graph I/O ===")
for inp in model.graph.input:
    print(f"  input: {inp.name}")
for out in model.graph.output:
    print(f"  output: {out.name}")
