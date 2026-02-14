"""Verify pre-transformer against PyTorch reference implementation.
Uses same codes as the C debug output to match exactly."""
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
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            if info["dtype"] == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(info["shape"])
            else:
                continue
            tensors[name] = arr
        return tensors

print("Loading safetensors...")
t = load_safetensors(MODEL)

def rms_norm(x, weight, eps=1e-6):
    """x: [T, D], weight: [D]"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

def rope_neox(x, T, heads, hd, theta=10000.0):
    """Apply rotary embeddings (neox style: split in half).
    x: [T, heads*hd], reshaped to [T, heads, hd]"""
    x = x.reshape(T, heads, hd)
    half = hd // 2
    freqs = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    positions = np.arange(T, dtype=np.float32)
    angles = np.outer(positions, freqs)  # [T, half]
    cos_a = np.cos(angles)  # [T, half]
    sin_a = np.sin(angles)  # [T, half]

    x1 = x[:, :, :half]   # first half
    x2 = x[:, :, half:]   # second half
    # neox: rotate pairs (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    cos_a = cos_a[:, None, :]  # [T, 1, half]
    sin_a = sin_a[:, None, :]
    r1 = x1 * cos_a - x2 * sin_a
    r2 = x1 * sin_a + x2 * cos_a
    return np.concatenate([r1, r2], axis=-1).reshape(T, heads * hd)

def causal_attention(Q, K, V, T, heads, hd):
    """Q, K, V: [T, heads*hd]. Returns [T, heads*hd]."""
    scale = 1.0 / np.sqrt(hd)
    Q = Q.reshape(T, heads, hd)
    K = K.reshape(T, heads, hd)
    V = V.reshape(T, heads, hd)

    out = np.zeros_like(Q)
    for h in range(heads):
        # [T, hd] @ [hd, T] = [T, T]
        scores = Q[:, h, :] @ K[:, h, :].T * scale
        # Causal mask
        mask = np.triu(np.full((T, T), -1e9), k=1)
        scores += mask
        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        out[:, h, :] = attn @ V[:, h, :]

    return out.reshape(T, heads * hd)

def silu(x):
    return x / (1.0 + np.exp(-x))

# Read the codes from C debug output
# Using the first run's codes (from the latest test)
codes_all = [
    [1342, 1557, 108, 81, 1468, 1276, 1671, 1653, 711, 24, 1852, 577, 439, 187, 266, 516],
    [1334, 684, 642, 1030, 238, 364, 1092, 242, 1542, 899, 1015, 1337, 853, 1618, 52, 1804],
]
T = len(codes_all)

# RVQ decode
def load_codebook(name_prefix):
    emb_sum = t[f"{name_prefix}._codebook.embedding_sum"]
    usage = t[f"{name_prefix}._codebook.cluster_usage"]
    return emb_sum / np.maximum(usage, 1e-7)[:, None]

cb0 = load_codebook("decoder.quantizer.rvq_first.vq.layers.0")
rest_cbs = [load_codebook(f"decoder.quantizer.rvq_rest.vq.layers.{i}") for i in range(15)]
proj_first = t["decoder.quantizer.rvq_first.output_proj.weight"].squeeze(-1)
proj_rest = t["decoder.quantizer.rvq_rest.output_proj.weight"].squeeze(-1)

rvq_out = np.zeros((512, T), dtype=np.float32)
for ti in range(T):
    for cb_idx in range(16):
        code = codes_all[ti][cb_idx]
        emb = cb0[code] if cb_idx == 0 else rest_cbs[cb_idx - 1][code]
        proj = proj_first if cb_idx == 0 else proj_rest
        rvq_out[:, ti] += proj @ emb

# Pre-conv
conv_w = t["decoder.pre_conv.conv.weight"]
conv_b = t["decoder.pre_conv.conv.bias"]
pad = 2
inp = np.pad(rvq_out, ((0, 0), (pad, 0)), mode='constant')
pre_conv = np.zeros((1024, T), dtype=np.float32)
for oc in range(1024):
    for ti in range(T):
        s = 0.0
        for ic in range(512):
            for ki in range(3):
                s += conv_w[oc, ic, ki] * inp[ic, ti + ki]
        pre_conv[oc, ti] = s + conv_b[oc]

print(f"pre_conv[0,:]: {' '.join(f'{v:.6f}' for v in pre_conv[0, :])}")

# Pre-transformer
input_proj_w = t["decoder.pre_transformer.input_proj.weight"]
input_proj_b = t.get("decoder.pre_transformer.input_proj.bias")
output_proj_w = t["decoder.pre_transformer.output_proj.weight"]
output_proj_b = t.get("decoder.pre_transformer.output_proj.bias")
final_norm_w = t["decoder.pre_transformer.norm.weight"]

# Transpose to [T, 1024]
x = pre_conv.T.copy()
# input_proj
x_h = x @ input_proj_w.T
if input_proj_b is not None:
    x_h += input_proj_b
print(f"input_proj[0,:5]: {' '.join(f'{v:.6f}' for v in x_h[0, :5])}")

for layer in range(8):
    ln_w = t[f"decoder.pre_transformer.layers.{layer}.input_layernorm.weight"]
    normed = rms_norm(x_h, ln_w)

    wq = t[f"decoder.pre_transformer.layers.{layer}.self_attn.q_proj.weight"]
    wk = t[f"decoder.pre_transformer.layers.{layer}.self_attn.k_proj.weight"]
    wv = t[f"decoder.pre_transformer.layers.{layer}.self_attn.v_proj.weight"]
    wo = t[f"decoder.pre_transformer.layers.{layer}.self_attn.o_proj.weight"]

    Q = normed @ wq.T
    K = normed @ wk.T
    V = normed @ wv.T

    if layer == 0:
        print(f"L0 Q[0,:5]: {' '.join(f'{v:.6f}' for v in Q[0, :5])}")

    Q = rope_neox(Q, T, 16, 64)
    K = rope_neox(K, T, 16, 64)

    attn_out = causal_attention(Q, K, V, T, 16, 64)
    proj = attn_out @ wo.T

    # LayerScale + residual
    attn_ls = t[f"decoder.pre_transformer.layers.{layer}.self_attn_layer_scale.scale"]
    proj *= attn_ls
    x_h = x_h + proj

    if layer == 0:
        print(f"L0 post-attn[0,:5]: {' '.join(f'{v:.6f}' for v in x_h[0, :5])}")

    # Post-attention norm
    post_ln_w = t[f"decoder.pre_transformer.layers.{layer}.post_attention_layernorm.weight"]
    normed2 = rms_norm(x_h, post_ln_w)

    # SwiGLU MLP
    gate_w = t[f"decoder.pre_transformer.layers.{layer}.mlp.gate_proj.weight"]
    up_w = t[f"decoder.pre_transformer.layers.{layer}.mlp.up_proj.weight"]
    down_w = t[f"decoder.pre_transformer.layers.{layer}.mlp.down_proj.weight"]

    gate = normed2 @ gate_w.T
    up = normed2 @ up_w.T
    ffn = silu(gate) * up
    ffn_out = ffn @ down_w.T

    mlp_ls = t[f"decoder.pre_transformer.layers.{layer}.mlp_layer_scale.scale"]
    ffn_out *= mlp_ls
    x_h = x_h + ffn_out

    print(f"Layer {layer} done: x_h[0,:5] = {' '.join(f'{v:.6f}' for v in x_h[0, :5])}")

# Final norm
x_h = rms_norm(x_h, final_norm_w)
# output_proj
x_out = x_h @ output_proj_w.T
if output_proj_b is not None:
    x_out += output_proj_b

# Transpose back to [1024, T]
xfmr_out = x_out.T

print(f"\nFinal xfmr[0,:5]: {' '.join(f'{v:.6f}' for v in xfmr_out[0, :])}")
print(f"Final xfmr[1,:5]: {' '.join(f'{v:.6f}' for v in xfmr_out[1, :])}")
