"""Verify post-transformer vocoder stages against C binary dumps.

Loads the xfmr output from C (voc_xfmr_out.raw), runs the remaining
stages (ConvNeXt upsample + BigVGAN decoder) in Python using the same
safetensors weights, and compares with C's stage outputs.
"""
import struct
import json
import numpy as np
import os
import sys
from math import erf as _erf

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")

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


def load_raw(fname, shape=None):
    if not os.path.exists(fname):
        return None
    arr = np.fromfile(fname, dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def compare(name, py, c_file, shape):
    c = load_raw(c_file, shape)
    if c is None:
        print(f"  {name}: C dump not found ({c_file})")
        return False
    if py.shape != c.shape:
        print(f"  {name}: SHAPE MISMATCH py={py.shape} c={c.shape}")
        return False
    diff = np.abs(py - c)
    max_diff = diff.max()
    rms_diff = np.sqrt(np.mean(diff**2))
    rms_py = np.sqrt(np.mean(py**2))
    corr = np.corrcoef(py.flatten()[:10000], c.flatten()[:10000])[0, 1] if py.size > 1 else 1.0
    ok = max_diff < 0.1
    status = "OK" if ok else "MISMATCH"
    print(f"  {name}: {status} (max_diff={max_diff:.6f}, rms_diff={rms_diff:.6f}, "
          f"rms={rms_py:.6f}, corr={corr:.6f})")
    if not ok:
        flat_py = py.flatten()
        flat_c = c.flatten()
        print(f"    py[:8]: {flat_py[:8]}")
        print(f"    c[:8]:  {flat_c[:8]}")
        idx = np.argmax(diff.flatten())
        print(f"    worst at [{idx}]: py={flat_py[idx]:.6f} c={flat_c[idx]:.6f}")
    return ok


# ========================================================================
# Vectorized Operations (numpy, no scipy)
# ========================================================================

def causal_conv1d(x, weight, bias, dilation=1, groups=1):
    """Causal Conv1d: x [C_in, T], weight [C_out, C_in/groups, K], bias [C_out].
    Uses im2col + matmul."""
    c_out, c_in_per_group, kernel = weight.shape
    c_in, T = x.shape
    pad = dilation * (kernel - 1)

    if groups > 1 and groups == c_in:
        # Depthwise: vectorized per kernel position
        padded = np.pad(x, ((0, 0), (pad, 0)), mode='constant')  # [C, T+pad]
        out = np.zeros((c_out, T), dtype=np.float32)
        for ki in range(kernel):
            w = weight[:, 0, ki]  # [C]
            out += w[:, None] * padded[:, ki * dilation:ki * dilation + T]
        if bias is not None:
            out += bias[:, None]
        return out

    # Standard conv1d: im2col
    padded = np.pad(x, ((0, 0), (pad, 0)), mode='constant')
    cols = np.zeros((c_in * kernel, T), dtype=np.float32)
    for ic in range(c_in):
        for ki in range(kernel):
            cols[ic * kernel + ki, :] = padded[ic, ki * dilation:ki * dilation + T]
    weight_flat = weight.reshape(c_out, c_in * kernel)
    out = weight_flat @ cols
    if bias is not None:
        out += bias[:, None]
    return out


def causal_conv_transpose1d(x, weight, bias, stride):
    """Causal ConvTranspose1d: x [C_in, T], weight [C_in, C_out, K].
    Vectorized using matmul per kernel position."""
    c_in, c_out, kernel = weight.shape
    _, T = x.shape
    T_full = (T - 1) * stride + kernel
    T_out = T * stride
    trim = kernel - stride

    out = np.zeros((c_out, T_full), dtype=np.float32)
    for ki in range(kernel):
        # W_ki[c_in, c_out] -> contribution = W_ki.T @ x -> [c_out, T]
        W_ki = weight[:, :, ki]  # [c_in, c_out]
        contrib = W_ki.T @ x    # [c_out, T]
        # Place at positions ki, ki+stride, ki+2*stride, ...
        out[:, ki::stride][:, :T] += contrib

    if trim > 0:
        out = out[:, :T_out]
    if bias is not None:
        out += bias[:, None]
    return out


def layer_norm_channels(x, weight, bias, eps=1e-5):
    """LayerNorm over channels for each time step. x: [C, T]"""
    mean = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std * weight[:, None] + bias[:, None]


def gelu_exact(x):
    """GELU exact, vectorized with numpy."""
    # Use the approximation via tanh (same as PyTorch default GELU)
    # Actually let's use the exact formula with numpy
    sqrt2 = np.float32(1.4142135623730951)
    # numpy erf is available
    return np.float32(0.5) * x * (np.float32(1.0) + np.vectorize(_erf)(x / sqrt2)).astype(np.float32)


def snake_beta(x, alpha_raw, beta_raw):
    """SnakeBeta: x + (1/exp(beta)) * sin^2(exp(alpha) * x)"""
    ea = np.exp(alpha_raw)[:, None]
    ieb = (1.0 / np.exp(beta_raw))[:, None]
    s = np.sin(ea * x)
    return x + ieb * s * s


# ========================================================================
# Main
# ========================================================================

print("Loading safetensors...")
t = load_safetensors(MODEL)
print("Loaded.")

xfmr_raw = load_raw("voc_xfmr_out.raw")
if xfmr_raw is None:
    print("ERROR: voc_xfmr_out.raw not found.")
    sys.exit(1)

T = len(xfmr_raw) // 1024
print(f"xfmr output: [1024, {T}]")
x = xfmr_raw.reshape(1024, T)

# ---- Upsample (2 stages, each 2x) ----
for stage in range(2):
    print(f"\n=== Upsample {stage} (T={T}) ===")
    tconv_w = t[f"decoder.upsample.{stage}.0.conv.weight"]
    tconv_b = t[f"decoder.upsample.{stage}.0.conv.bias"]
    x = causal_conv_transpose1d(x, tconv_w, tconv_b, stride=2)
    T *= 2
    print(f"  tconv -> [{x.shape[0]}, {T}], x[0,:5]={x[0,:5]}")

    dw_w = t[f"decoder.upsample.{stage}.1.dwconv.conv.weight"]
    dw_b = t[f"decoder.upsample.{stage}.1.dwconv.conv.bias"]
    norm_w = t[f"decoder.upsample.{stage}.1.norm.weight"]
    norm_b = t[f"decoder.upsample.{stage}.1.norm.bias"]
    pw1_w = t[f"decoder.upsample.{stage}.1.pwconv1.weight"]
    pw1_b = t[f"decoder.upsample.{stage}.1.pwconv1.bias"]
    pw2_w = t[f"decoder.upsample.{stage}.1.pwconv2.weight"]
    pw2_b = t[f"decoder.upsample.{stage}.1.pwconv2.bias"]
    gamma = t[f"decoder.upsample.{stage}.1.gamma"]

    residual = x.copy()
    x = causal_conv1d(x, dw_w, dw_b, groups=1024)
    x = layer_norm_channels(x, norm_w, norm_b)
    x = pw1_w @ x + pw1_b[:, None]
    x = gelu_exact(x)
    x = pw2_w @ x + pw2_b[:, None]
    x = gamma[:, None] * x
    x = residual + x
    print(f"  convnext -> x[0,:5]={x[0,:5]}")

    ok = compare(f"upsample{stage}", x, f"voc_upsample{stage}_out.raw", (1024, T))
    if not ok:
        sys.exit(1)

# ---- BigVGAN init ----
print(f"\n=== BigVGAN init (T={T}) ===")
x = causal_conv1d(x, t["decoder.decoder.0.conv.weight"], t["decoder.decoder.0.conv.bias"])
print(f"  init -> [{x.shape[0]}, {T}], x[0,:5]={x[0,:5]}")
ok = compare("bigvgan_init", x, "voc_bigvgan_init.raw", (1536, T))
if not ok:
    sys.exit(1)

# ---- BigVGAN blocks ----
rates = [8, 5, 4, 3]
channels = [1536, 768, 384, 192, 96]

for b in range(4):
    dec = b + 1
    in_ch, out_ch, rate = channels[b], channels[b + 1], rates[b]
    print(f"\n=== BigVGAN block {b} (rate={rate}, {in_ch}->{out_ch}, T={T}) ===")

    x = snake_beta(x, t[f"decoder.decoder.{dec}.block.0.alpha"],
                       t[f"decoder.decoder.{dec}.block.0.beta"])

    tconv_w = t[f"decoder.decoder.{dec}.block.1.conv.weight"]
    tconv_b = t[f"decoder.decoder.{dec}.block.1.conv.bias"]
    x = causal_conv_transpose1d(x, tconv_w, tconv_b, stride=rate)
    T *= rate
    print(f"  tconv -> [{x.shape[0]}, {T}], x[0,:5]={x[0,:5]}")

    for r in range(3):
        ru = r + 2
        dil = [1, 3, 9][r]
        res = x.copy()
        x = snake_beta(x, t[f"decoder.decoder.{dec}.block.{ru}.act1.alpha"],
                           t[f"decoder.decoder.{dec}.block.{ru}.act1.beta"])
        x = causal_conv1d(x, t[f"decoder.decoder.{dec}.block.{ru}.conv1.conv.weight"],
                              t[f"decoder.decoder.{dec}.block.{ru}.conv1.conv.bias"],
                          dilation=dil)
        x = snake_beta(x, t[f"decoder.decoder.{dec}.block.{ru}.act2.alpha"],
                           t[f"decoder.decoder.{dec}.block.{ru}.act2.beta"])
        c2_w = t[f"decoder.decoder.{dec}.block.{ru}.conv2.conv.weight"]
        c2_b = t[f"decoder.decoder.{dec}.block.{ru}.conv2.conv.bias"]
        x = c2_w[:, :, 0] @ x + c2_b[:, None]
        x = res + x
        print(f"  res{r} (dil={dil}): x[0,:5]={x[0,:5]}")

    ok = compare(f"bigvgan_blk{b}", x, f"voc_bigvgan_blk{b}.raw", (out_ch, T))
    if not ok:
        sys.exit(1)

# ---- Final ----
print(f"\n=== Final (T={T}) ===")
x = snake_beta(x, t["decoder.decoder.5.alpha"], t["decoder.decoder.5.beta"])
x = causal_conv1d(x, t["decoder.decoder.6.conv.weight"], t["decoder.decoder.6.conv.bias"])
x = np.clip(x, -1.0, 1.0)
print(f"  {x.shape[1]} samples, range [{x.min():.4f}, {x.max():.4f}]")
x.flatten().tofile("py_audio.raw")

onnx = load_raw("onnx_audio.raw")
if onnx is not None:
    py = x.flatten()[:len(onnx)]
    corr = np.corrcoef(py, onnx[:len(py)])[0, 1]
    rms_err = np.sqrt(np.mean((py - onnx[:len(py)])**2))
    rms_ref = np.sqrt(np.mean(onnx[:len(py)]**2))
    snr = 20 * np.log10(rms_ref / (rms_err + 1e-10))
    print(f"  Python vs ONNX: corr={corr:.6f}, rms={rms_err:.6f}, SNR={snr:.1f} dB")

native = load_raw("native_audio.raw")
if native is not None:
    py = x.flatten()[:len(native)]
    corr = np.corrcoef(py, native[:len(py)])[0, 1]
    rms_err = np.sqrt(np.mean((py - native[:len(py)])**2))
    print(f"  Python vs Native: corr={corr:.6f}, rms={rms_err:.6f}")
