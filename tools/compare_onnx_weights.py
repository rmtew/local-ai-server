"""Compare weights between ONNX model and safetensors to check if they're the same model."""
import struct
import json
import numpy as np
import os
import sys

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")

SAFETENSORS = r"C:\Data\R\git\claude-repos\deps\models\tts\Qwen3-TTS-Tokenizer-12Hz\model.safetensors"
ONNX_MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\qwen3-tts-12hz-0.6b-base\tokenizer12hz_decode.onnx"


def load_safetensors_header(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size)
        header = json.loads(header_json)
        return header


def load_safetensors_tensor(path, name):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size)
        header = json.loads(header_json)
        data_start = 8 + header_size
        info = header[name]
        begin, end = info["data_offsets"]
        f.seek(data_start + begin)
        raw = f.read(end - begin)
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(info["shape"])


# List safetensors weight names
print("=== Safetensors weights ===")
st_header = load_safetensors_header(SAFETENSORS)
st_names = sorted([n for n in st_header if n != "__metadata__"])
total_params = 0
for name in st_names:
    shape = st_header[name]["shape"]
    params = 1
    for s in shape:
        params *= s
    total_params += params
    print(f"  {name}: {shape} ({params:,} params)")
print(f"  Total: {total_params:,} params ({total_params * 4 / 1e6:.1f} MB)")

# Check ONNX model
print(f"\n=== ONNX model ===")
try:
    import onnx
    model = onnx.load(ONNX_MODEL, load_external_data=False)
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset: {[o.version for o in model.opset_import]}")

    # List initializers (embedded weights)
    initializers = model.graph.initializer
    print(f"  Initializers: {len(initializers)}")
    onnx_total = 0
    onnx_names = {}
    for init in initializers:
        shape = list(init.dims)
        dtype = init.data_type
        params = 1
        for s in shape:
            params *= s
        onnx_total += params
        onnx_names[init.name] = (shape, dtype, params)
        if params > 1000:
            print(f"  {init.name}: {shape} dtype={dtype} ({params:,})")
    print(f"  Total: {onnx_total:,} params")

    # Try to extract a specific weight and compare
    print(f"\n=== Weight comparison ===")
    # Find matching weights between ONNX and safetensors
    for init in initializers[:20]:
        name = init.name
        # Try to find matching safetensors weight
        # ONNX names might differ from safetensors names
        if name in st_header:
            st_w = load_safetensors_tensor(SAFETENSORS, name)
            onnx_w = np.array(onnx.numpy_helper.to_array(init))
            if st_w.shape == onnx_w.shape:
                diff = np.abs(st_w - onnx_w).max()
                print(f"  {name}: shapes match {st_w.shape}, max_diff={diff:.8f}")
            else:
                print(f"  {name}: shape mismatch st={st_w.shape} onnx={onnx_w.shape}")

except ImportError:
    print("  onnx module not available, trying onnxruntime inspection...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_MODEL)
        print(f"  Inputs: {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}")
        print(f"  Outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}")
    except Exception as e:
        print(f"  Error: {e}")

except Exception as e:
    print(f"  Error loading ONNX: {e}")
    # Try just reading the file header
    with open(ONNX_MODEL, "rb") as f:
        magic = f.read(4)
        print(f"  Magic bytes: {magic.hex()}")
