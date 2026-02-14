"""Run ONNX vocoder and compare with native Python implementation.

Uses onnxruntime to extract intermediate ONNX outputs at key stages,
then compares with the C/Python native vocoder outputs (from .raw dumps).
"""
import numpy as np
import onnxruntime as ort
import onnx
import struct, json, os, sys

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")

ONNX_MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\qwen3-tts-0.6b\tokenizer12hz_decode.onnx"
SAFETENSORS = r"C:\Data\R\git\claude-repos\deps\models\tts\Qwen3-TTS-Tokenizer-12Hz\model.safetensors"

# Load codes from the C debug output (parse the printed codes)
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
    row = [int(x) for x in line.split()]
    codes.append(row)
codes = np.array(codes, dtype=np.int64)
n_steps = codes.shape[0]
print(f"Codes: [{n_steps}, {codes.shape[1]}]")

# Run ONNX model
codes_input = codes.reshape(1, n_steps, 16)
print(f"ONNX input shape: {codes_input.shape}")

sess = ort.InferenceSession(ONNX_MODEL)
input_name = sess.get_inputs()[0].name
print(f"ONNX input name: {input_name}")
outputs = sess.run(None, {input_name: codes_input})
onnx_audio = outputs[0].flatten()
print(f"ONNX output: {len(onnx_audio)} elements")
print(f"ONNX audio[:10]: {onnx_audio[:10]}")
print(f"ONNX audio range: [{onnx_audio.min():.4f}, {onnx_audio.max():.4f}]")

# Compare with C native audio
native = np.fromfile("native_audio.raw", dtype=np.float32) if os.path.exists("native_audio.raw") else None
if native is not None:
    n = min(len(onnx_audio), len(native))
    onnx_trimmed = onnx_audio[:n]
    native_trimmed = native[:n]
    corr = np.corrcoef(onnx_trimmed, native_trimmed)[0, 1]
    rms_err = np.sqrt(np.mean((onnx_trimmed - native_trimmed)**2))
    print(f"\nONNX vs Native: corr={corr:.6f}, rms_err={rms_err:.6f}")
    print(f"  ONNX[:5] : {onnx_trimmed[:5]}")
    print(f"  Native[:5]: {native_trimmed[:5]}")

# Now let's also run the ONNX model with intermediate outputs
# Add all internal node outputs to the session
print("\n=== Extracting ONNX intermediate outputs ===")
model = onnx.load(ONNX_MODEL)

# Find nodes related to RVQ decode by looking at early computation
# Strategy: add intermediate outputs for key stage boundaries

# Find the output of the last ReduceSum (RVQ accumulation),
# and the output of the pre-conv
# We'll look for specific node patterns

# Let's find node outputs that match known shapes
# RVQ output should be [1, 512, T]
# Pre-conv output should be [1, 1024, T]
# Transformer output should be [1, 1024, T]

# Add ALL node outputs to get a trace (will be slow but informative)
# Let's be selective - look for Conv and ConvTranspose outputs
conv_outputs = []
for node in model.graph.node:
    if node.op_type in ("Conv", "ConvTranspose", "Add", "MatMul"):
        for out in node.output:
            conv_outputs.append(out)

# Too many. Let's just get the first Conv output (pre_conv) and compare
# Actually let's use a simpler approach: find specific named patterns
print("Looking for pre_conv and upsample boundaries...")

# Use the model with added outputs for key nodes
# Strategy: run with just a few intermediate outputs at a time

# Find the first Conv node (should be pre_conv)
first_convs = []
for i, node in enumerate(model.graph.node):
    if node.op_type == "Conv":
        first_convs.append((i, node.name, node.output[0], [str(x) for x in node.input]))
        if len(first_convs) > 5:
            break

print("\nFirst 5 Conv nodes:")
for idx, name, out, inp in first_convs:
    print(f"  [{idx}] {name or 'unnamed'}: output={out}, inputs={inp}")

# Find ConvTranspose nodes (upsample boundaries)
ctrans = []
for i, node in enumerate(model.graph.node):
    if node.op_type == "ConvTranspose":
        ctrans.append((i, node.name, node.output[0], [str(x) for x in node.input]))

print(f"\nConvTranspose nodes ({len(ctrans)}):")
for idx, name, out, inp in ctrans:
    print(f"  [{idx}] {name or 'unnamed'}: output={out}, inputs={inp}")

# Now add intermediate outputs for key boundaries and run
# Add output for the first Conv (likely pre_conv) output
model_with_outputs = onnx.ModelProto()
model_with_outputs.CopyFrom(model)

# Add intermediate outputs
outputs_to_add = []
# First conv output (pre_conv or early internal)
for idx, name, out, inp in first_convs[:3]:
    vi = onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, None)
    model_with_outputs.graph.output.append(vi)
    outputs_to_add.append(out)

# First ConvTranspose output (first upsample)
for idx, name, out, inp in ctrans[:2]:
    vi = onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, None)
    model_with_outputs.graph.output.append(vi)
    outputs_to_add.append(out)

print(f"\nRunning ONNX with {len(outputs_to_add)} intermediate outputs...")
sess2 = ort.InferenceSession(model_with_outputs.SerializeToString())
all_outputs = sess2.run(None, {input_name: codes_input})

print(f"Got {len(all_outputs)} outputs")
for i, (oname, oval) in enumerate(zip(
    [o.name for o in sess2.get_outputs()], all_outputs)):
    arr = np.array(oval)
    print(f"  [{i}] {oname}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")
    if arr.ndim >= 2:
        flat = arr.flatten()
        print(f"       first 5: {flat[:5]}")

# Compare first conv output with native pre_conv
# Load native xfmr output
xfmr_out = np.fromfile("voc_xfmr_out.raw", dtype=np.float32) if os.path.exists("voc_xfmr_out.raw") else None
if xfmr_out is not None:
    T = len(xfmr_out) // 1024
    xfmr_out = xfmr_out.reshape(1024, T)
    print(f"\nNative xfmr output: [1024, {T}]")
    print(f"  xfmr[0,:5]: {xfmr_out[0,:5]}")
