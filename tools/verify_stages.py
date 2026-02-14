"""Verify vocoder stages against native C implementation."""
import struct
import json
import numpy as np

MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\Qwen3-TTS-Tokenizer-12Hz\model.safetensors"

def load_safetensors(path):
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
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            if dtype_str == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(info["shape"])
            elif dtype_str == "BF16":
                raw16 = np.frombuffer(raw, dtype=np.uint16)
                raw32 = (raw16.astype(np.uint32) << 16)
                arr = raw32.view(np.float32).reshape(info["shape"]).copy()
            else:
                continue
            tensors[name] = arr
        return tensors

print("Loading safetensors...")
tensors = load_safetensors(MODEL)

# ======== RVQ Decode ========
codes_all = [
    [1221, 574, 373, 252, 1750, 736, 717, 1694, 1337, 386, 159, 1055, 1297, 1729, 1792, 370],
    [1342, 1198, 574, 415, 1621, 379, 334, 609, 16, 1141, 955, 218, 508, 677, 561, 1809],
]
T = len(codes_all)

def load_codebook(name_prefix):
    emb_sum = tensors[f"{name_prefix}._codebook.embedding_sum"]
    usage = tensors[f"{name_prefix}._codebook.cluster_usage"]
    return emb_sum / np.maximum(usage, 1e-7)[:, None]

cb0 = load_codebook("decoder.quantizer.rvq_first.vq.layers.0")
rest_cbs = [load_codebook(f"decoder.quantizer.rvq_rest.vq.layers.{i}") for i in range(15)]
proj_first = tensors["decoder.quantizer.rvq_first.output_proj.weight"].squeeze(-1)
proj_rest = tensors["decoder.quantizer.rvq_rest.output_proj.weight"].squeeze(-1)

rvq_out = np.zeros((512, T), dtype=np.float32)
for t in range(T):
    for cb_idx in range(16):
        code = codes_all[t][cb_idx]
        if cb_idx == 0:
            emb = cb0[code]
            proj = proj_first
        else:
            emb = rest_cbs[cb_idx - 1][code]
            proj = proj_rest
        rvq_out[:, t] += proj @ emb

print(f"RVQ[0,:5]: {' '.join(f'{v:.6f}' for v in rvq_out[0, :])}")
print(f"RVQ[1,:5]: {' '.join(f'{v:.6f}' for v in rvq_out[1, :])}")

# ======== Pre-conv: CausalConv1d(512, 1024, k=3) ========
conv_w = tensors["decoder.pre_conv.conv.weight"]  # [1024, 512, 3]
conv_b = tensors["decoder.pre_conv.conv.bias"]     # [1024]
print(f"\npre_conv weight shape: {conv_w.shape}")

# Causal: left-pad by (kernel-1)*dilation = 2
pad = 2
inp = np.pad(rvq_out, ((0, 0), (pad, 0)), mode='constant')  # [512, T+2]
print(f"padded input shape: {inp.shape}")

# Standard conv1d: out[oc, t] = sum_ic sum_ki w[oc, ic, ki] * inp[ic, t+ki]
pre_conv_out = np.zeros((1024, T), dtype=np.float32)
for oc in range(1024):
    for t in range(T):
        s = 0.0
        for ic in range(512):
            for ki in range(3):
                s += conv_w[oc, ic, ki] * inp[ic, t + ki]
        pre_conv_out[oc, t] = s + conv_b[oc]

print(f"\npre_conv[0,:5]: {' '.join(f'{v:.6f}' for v in pre_conv_out[0, :])}")
print(f"pre_conv[1,:5]: {' '.join(f'{v:.6f}' for v in pre_conv_out[1, :])}")

print(f"\nC output (from log):")
print(f"pre_conv[0,:5]: -0.025296 0.001599 -0.048316 -0.028441 0.038879")
print(f"pre_conv[1,:5]: 0.000643 0.000173 -0.000922 -0.000907 -0.000678")

# ======== Pre-transformer ========
# input_proj: [T, 1024] -> [T, 512]
input_proj_w = tensors["decoder.pre_transformer.input_proj.weight"]  # [512, 1024]
input_proj_b = tensors.get("decoder.pre_transformer.input_proj.bias")
print(f"\ninput_proj weight shape: {input_proj_w.shape}")
if input_proj_b is not None:
    print(f"input_proj bias shape: {input_proj_b.shape}")

# Transpose pre_conv_out to [T, 1024]
x = pre_conv_out.T  # [T, 1024]
# input_proj
x_h = x @ input_proj_w.T  # [T, 512]
if input_proj_b is not None:
    x_h += input_proj_b

print(f"\nAfter input_proj [T, 512]:")
print(f"  x_h[0,:5]: {' '.join(f'{v:.6f}' for v in x_h[0, :5])}")
print(f"  x_h[1,:5]: {' '.join(f'{v:.6f}' for v in x_h[1, :5])}")

# Layer 0 RMSNorm
ln_w = tensors["decoder.pre_transformer.layers.0.input_layernorm.weight"]  # [512]
eps = 1e-6
rms = np.sqrt(np.mean(x_h[0] ** 2) + eps)
normed = (x_h[0] / rms) * ln_w
print(f"\nAfter RMSNorm layer 0:")
print(f"  normed[0,:5]: {' '.join(f'{v:.6f}' for v in normed[:5])}")

# Q projection
wq = tensors["decoder.pre_transformer.layers.0.self_attn.q_proj.weight"]  # [1024, 512]
q = normed @ wq.T  # [1024]
print(f"\nQ[0,:5]: {' '.join(f'{v:.6f}' for v in q[:5])}")
