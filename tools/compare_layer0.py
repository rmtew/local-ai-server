"""Compare layer 0 intermediate values between Python and ONNX."""
import numpy as np
import onnx, onnxruntime as ort
import struct, json, os

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")
ONNX_MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\qwen3-tts-0.6b\tokenizer12hz_decode.onnx"
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

# Extract ONNX intermediate outputs for layer 0
model = onnx.load(ONNX_MODEL, load_external_data=False)
model2 = onnx.ModelProto()
model2.CopyFrom(model)

targets = [
    ('/decoder/pre_conv/conv/Conv_output_0', onnx.TensorProto.FLOAT),
    ('/decoder/pre_transformer/input_proj/Add_output_0', onnx.TensorProto.FLOAT),
    ('/decoder/pre_transformer/layers.0/input_layernorm/Mul_1_output_0', onnx.TensorProto.FLOAT),
    ('/decoder/pre_transformer/layers.0/self_attn/q_proj/MatMul_output_0', onnx.TensorProto.FLOAT),
    ('/decoder/pre_transformer/layers.0/self_attn/o_proj/MatMul_output_0', onnx.TensorProto.FLOAT),
    ('/decoder/pre_transformer/layers.0/Add_output_0', onnx.TensorProto.FLOAT),
]

for t, dtype in targets:
    vi = onnx.helper.make_tensor_value_info(t, dtype, None)
    model2.graph.output.append(vi)

print("Running ONNX with layer 0 intermediates...")
sess = ort.InferenceSession(model2.SerializeToString())
results = sess.run(None, {'audio_codes': codes_input})
output_names = [o.name for o in sess.get_outputs()]

onnx_vals = {}
for i, (name, val) in enumerate(zip(output_names, results)):
    for t, _ in targets:
        if name == t:
            onnx_vals[t] = np.array(val)

# Now run Python step by step for layer 0
prefix = 'decoder.pre_transformer.'
T = 18
H, AD, HEADS, HD = 512, 1024, 16, 64

preconv = onnx_vals['/decoder/pre_conv/conv/Conv_output_0']  # [1, 1024, 1024]
preconv_18 = preconv[0, :, :T]  # [1024, 18]
x = preconv_18.T  # [18, 1024]

# input_proj
input_proj_w = load_st(SAFETENSORS, prefix + 'input_proj.weight')
input_proj_b = load_st(SAFETENSORS, prefix + 'input_proj.bias')
py_input_proj = x @ input_proj_w.T + input_proj_b  # [18, 512]

# Compare input_proj
onnx_input_proj = onnx_vals['/decoder/pre_transformer/input_proj/Add_output_0']
print(f"\nONNX input_proj shape: {onnx_input_proj.shape}")  # probably [1, T, 512] or [1, 1024, 512]
onnx_ip = onnx_input_proj[0, :T, :]  # [18, 512]
print(f"input_proj: Py[0,:5]={py_input_proj[0,:5]}")
print(f"input_proj: OX[0,:5]={onnx_ip[0,:5]}")
d = np.abs(py_input_proj - onnx_ip)
print(f"input_proj diff: max={d.max():.10f}, mean={d.mean():.10f}")

# Layer 0 input_layernorm
lp = prefix + 'layers.0.'
in_norm_w = load_st(SAFETENSORS, lp + 'input_layernorm.weight')
py_normed = rms_norm(py_input_proj, in_norm_w)

onnx_normed = onnx_vals['/decoder/pre_transformer/layers.0/input_layernorm/Mul_1_output_0']
onnx_nm = onnx_normed[0, :T, :]
print(f"\nrms_norm: Py[0,:5]={py_normed[0,:5]}")
print(f"rms_norm: OX[0,:5]={onnx_nm[0,:5]}")
d = np.abs(py_normed - onnx_nm)
print(f"rms_norm diff: max={d.max():.10f}, mean={d.mean():.10f}")

# Q projection
wq = load_st(SAFETENSORS, lp + 'self_attn.q_proj.weight')  # [1024, 512]
py_Q = py_normed @ wq.T  # [18, 1024]

onnx_Q = onnx_vals['/decoder/pre_transformer/layers.0/self_attn/q_proj/MatMul_output_0']
onnx_q = onnx_Q[0, :T, :]
print(f"\nQ proj: Py[0,:5]={py_Q[0,:5]}")
print(f"Q proj: OX[0,:5]={onnx_q[0,:5]}")
d = np.abs(py_Q - onnx_q)
print(f"Q proj diff: max={d.max():.10f}, mean={d.mean():.10f}")

# O proj output
onnx_O = onnx_vals['/decoder/pre_transformer/layers.0/self_attn/o_proj/MatMul_output_0']
onnx_o = onnx_O[0, :T, :]
print(f"\nO proj: OX[0,:5]={onnx_o[0,:5]}")

# After layer 0 residual
onnx_after_l0 = onnx_vals['/decoder/pre_transformer/layers.0/Add_output_0']
onnx_al0 = onnx_after_l0[0, :T, :]

# Compute Python attention for layer 0
wk = load_st(SAFETENSORS, lp + 'self_attn.k_proj.weight')
wv = load_st(SAFETENSORS, lp + 'self_attn.v_proj.weight')
wo = load_st(SAFETENSORS, lp + 'self_attn.o_proj.weight')
attn_ls = load_st(SAFETENSORS, lp + 'self_attn_layer_scale.scale')

K = py_normed @ wk.T
V = py_normed @ wv.T
Q_r = py_Q.reshape(T, HEADS, HD)
K_r = K.reshape(T, HEADS, HD)
V_r = V.reshape(T, HEADS, HD)
Q_r = rope_neox(Q_r, T, HD)
K_r = rope_neox(K_r, T, HD)

scale = 1.0 / np.sqrt(float(HD))
attn_out = np.zeros_like(Q_r)
for h in range(HEADS):
    scores = Q_r[:, h, :] @ K_r[:, h, :].T * scale
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
    scores += mask
    sm = scores.max(axis=-1, keepdims=True)
    es = np.exp(scores - sm)
    aw = es / es.sum(axis=-1, keepdims=True)
    attn_out[:, h, :] = aw @ V_r[:, h, :]

py_o = attn_out.reshape(T, AD) @ wo.T
print(f"O proj: Py[0,:5]={py_o[0,:5]}")
d = np.abs(py_o - onnx_o)
print(f"O proj diff: max={d.max():.10f}, mean={d.mean():.10f}")

# After layer scale + residual
py_o_scaled = py_o * attn_ls
py_after_l0 = py_input_proj + py_o_scaled
print(f"\nAfter L0: Py[0,:5]={py_after_l0[0,:5]}")
print(f"After L0: OX[0,:5]={onnx_al0[0,:5]}")
d = np.abs(py_after_l0 - onnx_al0)
print(f"After L0 diff: max={d.max():.10f}, mean={d.mean():.10f}")
for t in range(min(5, T)):
    dt = np.abs(py_after_l0[t] - onnx_al0[t]).max()
    print(f"  t={t}: max={dt:.10f}")
