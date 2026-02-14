"""Verify RVQ decode against native C implementation."""
import struct
import json
import numpy as np

# Path to the safetensors model
MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\Qwen3-TTS-Tokenizer-12Hz\model.safetensors"

def load_safetensors(path):
    """Load safetensors file, return dict of name -> numpy array."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size)
        header = json.loads(header_json)
        data_start = 8 + header_size

        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype_str = info["dtype"]
            shape = info["data_offsets"]
            begin, end = info["data_offsets"]

            f.seek(data_start + begin)
            raw = f.read(end - begin)

            if dtype_str == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])
            elif dtype_str == "BF16":
                # Convert bf16 to f32
                raw16 = np.frombuffer(raw, dtype=np.uint16)
                raw32 = (raw16.astype(np.uint32) << 16)
                arr = raw32.view(np.float32).reshape(info["shape"])
            else:
                print(f"  Skipping {name}: dtype={dtype_str}")
                continue
            tensors[name] = arr
        return tensors

print("Loading safetensors...")
tensors = load_safetensors(MODEL)

# First time step codes from the C output
codes_t0 = [1221, 574, 373, 252, 1750, 736, 717, 1694, 1337, 386, 159, 1055, 1297, 1729, 1792, 370]
codes_t1 = [1342, 1198, 574, 415, 1621, 379, 334, 609, 16, 1141, 955, 218, 508, 677, 561, 1809]

# Load codebook embeddings and normalize
def load_codebook(name_prefix):
    emb_sum = tensors[f"{name_prefix}._codebook.embedding_sum"]
    usage = tensors[f"{name_prefix}._codebook.cluster_usage"]
    usage_clamped = np.maximum(usage, 1e-7)
    return emb_sum / usage_clamped[:, None]

# Load all 16 codebooks
cb0 = load_codebook("decoder.quantizer.rvq_first.vq.layers.0")
print(f"Codebook 0 shape: {cb0.shape}")  # [2048, 256]

rest_cbs = []
for i in range(15):
    cb = load_codebook(f"decoder.quantizer.rvq_rest.vq.layers.{i}")
    rest_cbs.append(cb)

# Load projection weights
proj_first = tensors["decoder.quantizer.rvq_first.output_proj.weight"]  # [512, 256, 1]
proj_rest = tensors["decoder.quantizer.rvq_rest.output_proj.weight"]    # [512, 256, 1]
print(f"proj_first shape: {proj_first.shape}")
print(f"proj_rest shape: {proj_rest.shape}")

# Squeeze the kernel dim
proj_first = proj_first.squeeze(-1)  # [512, 256]
proj_rest = proj_rest.squeeze(-1)    # [512, 256]

# Compute RVQ decode for t=0
T = 2
codes_all = [codes_t0, codes_t1]
result = np.zeros((512, T), dtype=np.float32)

for t in range(T):
    codes = codes_all[t]
    for cb_idx in range(16):
        code = codes[cb_idx]
        if cb_idx == 0:
            emb = cb0[code]  # [256]
            proj = proj_first  # [512, 256]
        else:
            emb = rest_cbs[cb_idx - 1][code]  # [256]
            proj = proj_rest  # [512, 256]

        projected = proj @ emb  # [512]
        result[:, t] += projected

print(f"\nRVQ output [512, {T}]:")
print(f"  RVQ[0,:5]: {' '.join(f'{v:.6f}' for v in result[0, :5])}")
print(f"  RVQ[1,:5]: {' '.join(f'{v:.6f}' for v in result[1, :5])}")

# C output for comparison:
print(f"\nC output (from log):")
print(f"  RVQ[0,:5]: 22.944506 1.918322 14.043383 12.237225 15.794068")
print(f"  RVQ[1,:5]: -1.760452 -6.606938 -6.251305 3.447453 10.171865")
