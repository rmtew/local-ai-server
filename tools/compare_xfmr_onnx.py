"""Extract transformer output from ONNX and compare with native."""
import numpy as np
import onnxruntime as ort
import onnx
import os

os.chdir(r"C:\Data\R\git\claude-repos\local-ai-server")
ONNX_MODEL = r"C:\Data\R\git\claude-repos\deps\models\tts\qwen3-tts-0.6b\tokenizer12hz_decode.onnx"

# Same codes as before
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

# Load ONNX model and add intermediate outputs
model = onnx.load(ONNX_MODEL, load_external_data=False)

# Key outputs to extract:
# 1. Pre-conv output: /decoder/pre_conv/conv/Conv_output_0
# 2. Transformer output (input to first upsample): /decoder/Transpose_19_output_0
# 3. First upsample convnext output (input to second upsample): /decoder/upsample.0.1/Add_output_0
# 4. Second upsample output: /decoder/upsample.1.0/conv/ConvTranspose_output_0 (already shown)

# Also look for RVQ decode output
# The RVQ first codebook output_proj produces: /decoder/output_proj/Conv_output_0
# The RVQ rest codebooks output_proj: /decoder/output_proj_1/Conv_output_0
# The sum of all codebook contributions should be somewhere

# Let me find the Add node that sums RVQ contributions
rvq_sum_outputs = []
for node in model.graph.node:
    for out in node.output:
        # Look for the transformer input
        if "Transpose_19" in out:
            rvq_sum_outputs.append(out)
        # Also look for the Add that combines RVQ first + rest
        if "ReduceSum" in out or ("Add" in out and "decoder" in out):
            pass  # too many

# Find specific named outputs
targets = [
    "/decoder/pre_conv/conv/Conv_output_0",
    "/decoder/Transpose_19_output_0",
    "/decoder/upsample.0.1/Add_output_0",
]

# Find the node that takes the pre_conv output as input to the transformer
# The transformer input is pre_conv output -> transpose -> input_proj
# Let's find what feeds into Transpose_19
for node in model.graph.node:
    if any("Transpose_19" in o for o in node.output):
        print(f"Transpose_19 node: op={node.op_type}, inputs={list(node.input)}, output={list(node.output)}")
        # Find the node that produces the input
        xfmr_input_name = node.input[0]
        targets.append(xfmr_input_name)
        break

# Add all target intermediate outputs
model2 = onnx.ModelProto()
model2.CopyFrom(model)
for t in targets:
    vi = onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
    model2.graph.output.append(vi)

print(f"\nRunning ONNX with {len(targets)} intermediate outputs...")
sess = ort.InferenceSession(model2.SerializeToString())
results = sess.run(None, {"audio_codes": codes_input})

output_names = [o.name for o in sess.get_outputs()]
print(f"Got {len(results)} outputs\n")

for i, (name, val) in enumerate(zip(output_names, results)):
    arr = np.array(val)
    print(f"[{i}] {name}: shape={arr.shape}")
    if arr.ndim >= 3:
        # For [1, C, T] format
        C = arr.shape[1]
        T = arr.shape[2] if arr.ndim > 2 else 1
        # Show first channel, first T values (up to 5)
        first = arr[0, 0, :min(5, T)] if T > 1 else arr[0, 0]
        print(f"    [0,0,:5]: {first}")
        first2 = arr[0, 1, :min(5, T)] if C > 1 and T > 1 else None
        if first2 is not None:
            print(f"    [0,1,:5]: {first2}")
        # Show last valid positions (around t=17)
        if T >= 18:
            print(f"    [0,0,16:20]: {arr[0, 0, 16:20]}")
            print(f"    [0,0,17:21]: {arr[0, 0, 17:21]}")

# Now compare transformer output with native
native_xfmr = np.fromfile("voc_xfmr_out.raw", dtype=np.float32)
T_native = len(native_xfmr) // 1024
native_xfmr = native_xfmr.reshape(1024, T_native)
print(f"\nNative xfmr output: [1024, {T_native}]")
print(f"  native[0,:5]: {native_xfmr[0,:5]}")
print(f"  native[1,:5]: {native_xfmr[1,:5]}")

# Find the ONNX transformer output in results
for i, name in enumerate(output_names):
    if "Transpose_19" in name:
        onnx_xfmr = np.array(results[i])  # shape (1, 1024, T_onnx)
        print(f"\nONNX xfmr output: {onnx_xfmr.shape}")
        onnx_first18 = onnx_xfmr[0, :, :18]  # [1024, 18]
        print(f"  onnx[0,:5]: {onnx_first18[0,:5]}")
        print(f"  onnx[1,:5]: {onnx_first18[1,:5]}")

        # Compare
        diff = np.abs(onnx_first18 - native_xfmr)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"\n  ONNX xfmr vs Native xfmr (first 18 timesteps):")
        print(f"    max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
        corr = np.corrcoef(onnx_first18.flatten(), native_xfmr.flatten())[0,1]
        print(f"    correlation={corr:.8f}")

        # Show position-by-position max diff
        for t in range(min(18, onnx_first18.shape[1])):
            d = np.abs(onnx_first18[:, t] - native_xfmr[:, t]).max()
            print(f"    t={t}: max_diff={d:.8f}")
        break

# Also compare upsample0 output
native_up0 = np.fromfile("voc_upsample0_out.raw", dtype=np.float32)
T_up0 = len(native_up0) // 1024
native_up0 = native_up0.reshape(1024, T_up0)
print(f"\nNative upsample0: [1024, {T_up0}]")
for i, name in enumerate(output_names):
    if "upsample.0.1/Add" in name:
        onnx_up0 = np.array(results[i])
        print(f"ONNX upsample0 after convnext: {onnx_up0.shape}")
        onnx_first = onnx_up0[0, :, :T_up0]
        diff = np.abs(onnx_first - native_up0)
        print(f"  max_diff={diff.max():.8f}, mean_diff={diff.mean():.8f}")
        corr = np.corrcoef(onnx_first.flatten(), native_up0.flatten())[0,1]
        print(f"  correlation={corr:.8f}")
        for t in range(min(10, T_up0)):
            d = np.abs(onnx_first[:, t] - native_up0[:, t]).max()
            print(f"  t={t}: max_diff={d:.8f}")
        break
