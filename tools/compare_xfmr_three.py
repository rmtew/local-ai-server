"""Run Python transformer on ONNX pre_conv output and compare all three."""
import numpy as np
import onnx, onnxruntime as ort
import struct, json, os

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")
ONNX_MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\qwen3-tts-12hz-0.6b-base\tokenizer12hz_decode.onnx"
SAFETENSORS = r"C:\Data\R\git\claude-repos\deps\models\tts\Qwen3-TTS-Tokenizer-12Hz\model.safetensors"

codes_lines = """1995 844 1357 1690 2028 1626 1579 1827 1179 1124 1868 1685 1273 856 177 1635
215 1863 1442 1088 405 292 504 658 1012 11 59 1726 32 1514 1202 552
1925 454 236 315 1822 736 777 678 1884 652 691 45 1868 17 176 3
1148 1270 152 324 159 878 142 678 1337 1025 257 831 1868 465 554 266
11 1081 236 288 1223 138 882 87 1887 51 155 262 271 309 1165 277
363 1259 1974 825 1822 892 294 1057 1458 818 644 1383 439 1937 753 47
1565 843 26 298 1625 1322 722 194 28 239 469 1855 31 1175 971 310
140 553 1437 533 144 583 1305 53 426 1328 158 104 120 171 826 580
490 652 302 1705 628 610 969 1879 1589 1315 123 909 1997 971 827 370
680 428 32 1460 628 288 416 368 1569 120 1503 256 350 1859 2019 534
1121 605 1049 1746 595 485 753 235 528 1532 548 1932 464 552 547 800
1235 170 665 438 628 217 450 133 799 851 715 16 1111 979 1795 252
56 685 1390 1107 1599 92 515 418 196 656 1521 1462 1268 1013 661 232
56 1276 1731 1580 1162 445 378 1017 1046 546 468 1641 1810 1692 213 183
106 336 1317 1643 1919 888 331 2033 665 884 515 1860 1597 654 1985 774
1047 844 1911 128 1651 1578 872 1584 170 486 1227 1973 1318 1445 100 544
467 724 1277 1893 846 1883 1567 1551 842 311 1577 222 802 17 117 558
302 614 1666 1931 565 1475 103 1275 839 110 37 738 344 559 25 381"""

codes = []
for line in codes_lines.strip().split('\n'):
    codes.append([int(x) for x in line.split()])
codes_input = np.array(codes, dtype=np.int64).reshape(1, 18, 16)

def load_st(path, name):
    with open(path, 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hs))
        ds = 8 + hs
        info = hdr[name]
        b, e = info['data_offsets']
        f.seek(ds + b)
        return np.frombuffer(f.read(e - b), dtype=np.float32).copy().reshape(info['shape'])

def rms_norm(x, w, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms * w

def rope_neox(x, T, head_dim, theta=10000.0):
    half = head_dim // 2
    freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    pos = np.arange(T, dtype=np.float32)
    angles = np.outer(pos, freq)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    x1 = x[:, :, :half]
    x2 = x[:, :, half:]
    rx1 = x1 * cos_a[:, None, :] - x2 * sin_a[:, None, :]
    rx2 = x2 * cos_a[:, None, :] + x1 * sin_a[:, None, :]
    return np.concatenate([rx1, rx2], axis=-1)

def run_transformer_python(preconv_1024_T, T):
    """Run 8-layer transformer in Python. preconv_1024_T is [1024, T] channels-first."""
    prefix = 'decoder.pre_transformer.'
    H, AD, HEADS, HD = 512, 1024, 16, 64

    # Transpose to [T, 1024]
    x = preconv_1024_T.T

    # input_proj
    input_proj_w = load_st(SAFETENSORS, prefix + 'input_proj.weight')
    input_proj_b = load_st(SAFETENSORS, prefix + 'input_proj.bias')
    x_h = x @ input_proj_w.T + input_proj_b
    print(f'  input_proj[0,:5]: {x_h[0,:5]}')

    for layer in range(8):
        lp = prefix + f'layers.{layer}.'
        in_norm = load_st(SAFETENSORS, lp + 'input_layernorm.weight')
        wq = load_st(SAFETENSORS, lp + 'self_attn.q_proj.weight')
        wk = load_st(SAFETENSORS, lp + 'self_attn.k_proj.weight')
        wv = load_st(SAFETENSORS, lp + 'self_attn.v_proj.weight')
        wo = load_st(SAFETENSORS, lp + 'self_attn.o_proj.weight')
        attn_ls = load_st(SAFETENSORS, lp + 'self_attn_layer_scale.scale')
        post_norm = load_st(SAFETENSORS, lp + 'post_attention_layernorm.weight')
        gate_w = load_st(SAFETENSORS, lp + 'mlp.gate_proj.weight')
        up_w = load_st(SAFETENSORS, lp + 'mlp.up_proj.weight')
        down_w = load_st(SAFETENSORS, lp + 'mlp.down_proj.weight')
        mlp_ls = load_st(SAFETENSORS, lp + 'mlp_layer_scale.scale')

        normed = rms_norm(x_h, in_norm)
        Q = (normed @ wq.T).reshape(T, HEADS, HD)
        K = (normed @ wk.T).reshape(T, HEADS, HD)
        V = (normed @ wv.T).reshape(T, HEADS, HD)

        Q = rope_neox(Q, T, HD)
        K = rope_neox(K, T, HD)

        scale = 1.0 / np.sqrt(float(HD))
        attn_out = np.zeros_like(Q)
        for h in range(HEADS):
            scores = Q[:, h, :] @ K[:, h, :].T * scale
            mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
            scores += mask
            scores_max = scores.max(axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
            attn_out[:, h, :] = attn_weights @ V[:, h, :]

        o_out = attn_out.reshape(T, AD) @ wo.T
        o_out *= attn_ls
        x_h = x_h + o_out

        normed2 = rms_norm(x_h, post_norm)
        gate = normed2 @ gate_w.T
        up = normed2 @ up_w.T
        silu_gate = gate / (1.0 + np.exp(-gate))
        ffn = silu_gate * up
        down = ffn @ down_w.T
        down *= mlp_ls
        x_h = x_h + down

    norm_w = load_st(SAFETENSORS, prefix + 'norm.weight')
    x_h = rms_norm(x_h, norm_w)
    output_proj_w = load_st(SAFETENSORS, prefix + 'output_proj.weight')
    output_proj_b = load_st(SAFETENSORS, prefix + 'output_proj.bias')
    x_out = x_h @ output_proj_w.T + output_proj_b
    return x_out.T  # [1024, T]


# Get ONNX pre_conv and transformer outputs
model = onnx.load(ONNX_MODEL, load_external_data=False)
model2 = onnx.ModelProto()
model2.CopyFrom(model)
for t in ['/decoder/pre_conv/conv/Conv_output_0', '/decoder/Transpose_19_output_0']:
    vi = onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
    model2.graph.output.append(vi)
sess = ort.InferenceSession(model2.SerializeToString())
results = sess.run(None, {'audio_codes': codes_input})
output_names = [o.name for o in sess.get_outputs()]

onnx_preconv = onnx_xfmr = None
for i, name in enumerate(output_names):
    if 'pre_conv' in name:
        onnx_preconv = results[i]
    if 'Transpose_19' in name:
        onnx_xfmr = results[i]

onnx_preconv_18 = onnx_preconv[0, :, :18]  # [1024, 18]
onnx_xfmr_18 = onnx_xfmr[0, :, :18]       # [1024, 18]

print(f"ONNX pre_conv: {onnx_preconv.shape}, xfmr: {onnx_xfmr.shape}")

# Load native transformer output
native_xfmr = np.fromfile('voc_xfmr_out.raw', dtype=np.float32).reshape(1024, 18)

# Run Python transformer on ONNX pre_conv (18 timesteps)
print("\n=== Python transformer (18 steps, using ONNX pre_conv) ===")
py_xfmr = run_transformer_python(onnx_preconv_18, 18)

print(f"\nPython xfmr[0,:5]: {py_xfmr[0,:5]}")
print(f"Native xfmr[0,:5]: {native_xfmr[0,:5]}")
print(f"ONNX   xfmr[0,:5]: {onnx_xfmr_18[0,:5]}")

d_pn = np.abs(py_xfmr - native_xfmr)
d_po = np.abs(py_xfmr - onnx_xfmr_18)
d_no = np.abs(native_xfmr - onnx_xfmr_18)
print(f"\nPython vs Native: max={d_pn.max():.8f}, mean={d_pn.mean():.8f}")
print(f"Python vs ONNX:   max={d_po.max():.8f}, mean={d_po.mean():.8f}")
print(f"Native vs ONNX:   max={d_no.max():.8f}, mean={d_no.mean():.8f}")

# Per-timestep
print("\nPer-timestep max diff:")
for t in range(18):
    d1 = np.abs(py_xfmr[:, t] - onnx_xfmr_18[:, t]).max()
    d2 = np.abs(native_xfmr[:, t] - onnx_xfmr_18[:, t]).max()
    d3 = np.abs(py_xfmr[:, t] - native_xfmr[:, t]).max()
    print(f"  t={t:2d}: Py-ONNX={d1:.8f}  Native-ONNX={d2:.8f}  Py-Native={d3:.8f}")
