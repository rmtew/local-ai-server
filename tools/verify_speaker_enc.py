"""Verify ECAPA-TDNN speaker encoder against C implementation.

Loads speaker_encoder.* weights from model.safetensors, runs the full
forward pass in numpy (matching tts_speaker_enc.c), and validates with
a deterministic mel input. Compares against C dump if available.

Fully headless -- no server, no pytorch, only numpy.

Usage:
    python tools/verify_speaker_enc.py                     # default model path
    python tools/verify_speaker_enc.py path/to/model.safetensors
"""
import numpy as np
import struct
import json
import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ---- Safetensors loader (handles BF16 and F32) ----

def load_safetensors(path):
    """Load all tensors from safetensors file, converting BF16 to F32."""
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
            dtype = info["dtype"]
            shape = info["shape"]
            if dtype == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
            elif dtype == "BF16":
                raw16 = np.frombuffer(raw, dtype=np.uint16).copy()
                raw32 = raw16.astype(np.uint32) << 16
                arr = raw32.view(np.float32).reshape(shape)
            else:
                continue
            tensors[name] = arr
        return tensors


def find_model_path():
    """Find model.safetensors in common locations."""
    deps = os.environ.get("DEPS_ROOT", "")
    candidates = [
        os.path.join(deps, "models", "tts", "qwen3-tts-12hz-0.6b-base", "model.safetensors"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ---- Conv1d operations (matching tts_speaker_enc.c) ----

def conv1d_same(x, weight, bias, dilation=1):
    """Same-padded Conv1d. x: [c_in, T], weight: [c_out, c_in, K]."""
    c_out, c_in, K = weight.shape
    T = x.shape[1]
    pad = dilation * (K - 1) // 2

    # Zero-pad both sides
    padded = np.pad(x, ((0, 0), (pad, pad)), mode='constant')

    # im2col
    cols = np.zeros((c_in * K, T), dtype=np.float32)
    for ic in range(c_in):
        for ki in range(K):
            cols[ic * K + ki, :] = padded[ic, ki * dilation:ki * dilation + T]

    out = weight.reshape(c_out, c_in * K) @ cols
    if bias is not None:
        out += bias[:, None]
    return out


def squeeze_k1(w):
    """Squeeze trailing dim from k=1 conv weights [c_out, c_in, 1] -> [c_out, c_in]."""
    return w.squeeze(2) if w.ndim == 3 else w

def conv1d_k1(x, weight, bias):
    """Conv1d with kernel=1 (linear projection). x: [c_in, T], weight: [c_out, c_in]."""
    weight = squeeze_k1(weight)
    out = weight @ x
    if bias is not None:
        out += bias[:, None]
    return out


def relu(x):
    return np.maximum(x, 0)


# ---- ECAPA-TDNN forward pass ----

def seres2net_forward(x, blk_weights, dilation):
    """SE-Res2Net block. x: [512, T] -> [512, T]."""
    C = 512
    S = 8  # scale
    G = C // S  # 64
    T = x.shape[1]

    residual = x.copy()

    # tdnn1: Conv1d(512->512, k=1) + ReLU
    x = relu(conv1d_k1(x, blk_weights["tdnn1_w"], blk_weights["tdnn1_b"]))

    # Res2Net: split into 8 groups of 64
    groups = [x[g * G:(g + 1) * G, :] for g in range(S)]
    prev_out = np.zeros((G, T), dtype=np.float32)

    for g in range(1, S):
        group_in = groups[g] + prev_out
        groups[g] = relu(conv1d_same(
            group_in,
            blk_weights[f"res{g - 1}_w"],
            blk_weights[f"res{g - 1}_b"],
            dilation=dilation
        ))
        prev_out = groups[g].copy()

    x = np.concatenate(groups, axis=0)  # [512, T]

    # tdnn2: Conv1d(512->512, k=1) + ReLU
    x = relu(conv1d_k1(x, blk_weights["tdnn2_w"], blk_weights["tdnn2_b"]))

    # SE: Squeeze-and-Excitation
    se_pool = x.mean(axis=1)  # [512]
    se_fc1_w = squeeze_k1(blk_weights["se_fc1_w"])
    se_fc2_w = squeeze_k1(blk_weights["se_fc2_w"])
    se_h = relu(se_fc1_w @ se_pool + blk_weights["se_fc1_b"])  # [128]
    se_s = 1.0 / (1.0 + np.exp(-(se_fc2_w @ se_h + blk_weights["se_fc2_b"])))  # [512]
    x = x * se_s[:, None]

    # Residual
    x = x + residual
    return x


def ecapa_tdnn_forward(tensors, mel, n_frames):
    """Full ECAPA-TDNN forward pass. mel: [128, T] -> embedding: [1024]."""
    T = n_frames

    # Block 0: Conv1d(128->512, k=5, same_pad=2) + ReLU
    w = tensors["speaker_encoder.blocks.0.conv.weight"]
    b = tensors["speaker_encoder.blocks.0.conv.bias"]
    x = relu(conv1d_same(mel, w, b, dilation=1))
    assert x.shape == (512, T), f"block0: expected (512, {T}), got {x.shape}"

    # Blocks 1-3: SE-Res2Net
    dilations = [2, 3, 4]
    block_outs = []

    for bi in range(3):
        idx = bi + 1
        d = dilations[bi]
        blk_w = {
            "tdnn1_w": tensors[f"speaker_encoder.blocks.{idx}.tdnn1.conv.weight"],
            "tdnn1_b": tensors[f"speaker_encoder.blocks.{idx}.tdnn1.conv.bias"],
            "tdnn2_w": tensors[f"speaker_encoder.blocks.{idx}.tdnn2.conv.weight"],
            "tdnn2_b": tensors[f"speaker_encoder.blocks.{idx}.tdnn2.conv.bias"],
            "se_fc1_w": tensors[f"speaker_encoder.blocks.{idx}.se_block.conv1.weight"],
            "se_fc1_b": tensors[f"speaker_encoder.blocks.{idx}.se_block.conv1.bias"],
            "se_fc2_w": tensors[f"speaker_encoder.blocks.{idx}.se_block.conv2.weight"],
            "se_fc2_b": tensors[f"speaker_encoder.blocks.{idx}.se_block.conv2.bias"],
        }
        for g in range(7):
            blk_w[f"res{g}_w"] = tensors[f"speaker_encoder.blocks.{idx}.res2net_block.blocks.{g}.conv.weight"]
            blk_w[f"res{g}_b"] = tensors[f"speaker_encoder.blocks.{idx}.res2net_block.blocks.{g}.conv.bias"]

        x = seres2net_forward(x, blk_w, d)
        block_outs.append(x.copy())

    # MFA: concat(block1, block2, block3) -> [1536, T]
    mfa_concat = np.concatenate(block_outs, axis=0)
    assert mfa_concat.shape == (1536, T)

    # MFA conv: Conv1d(1536->1536, k=1) + ReLU
    # k=1 weights stored as [c_out, c_in, 1] -- squeeze to [c_out, c_in]
    mfa_w = tensors["speaker_encoder.mfa.conv.weight"]
    if mfa_w.ndim == 3:
        mfa_w = mfa_w.squeeze(2)
    mfa_b = tensors["speaker_encoder.mfa.conv.bias"]
    hidden = relu(conv1d_k1(mfa_concat, mfa_w, mfa_b))

    # ASP: Attentive Statistics Pooling
    mean_vec = hidden.mean(axis=1)  # [1536]
    std_vec = np.sqrt(hidden.var(axis=1) + 1e-5)  # [1536]

    # concat(hidden, mean_broadcast, std_broadcast) -> [4608, T]
    asp_input = np.concatenate([
        hidden,
        np.tile(mean_vec[:, None], (1, T)),
        np.tile(std_vec[:, None], (1, T)),
    ], axis=0)
    assert asp_input.shape == (4608, T)

    # conv1: [4608->128, k=1] + ReLU + Tanh
    asp_c1_w = tensors["speaker_encoder.asp.tdnn.conv.weight"]
    if asp_c1_w.ndim == 3:
        asp_c1_w = asp_c1_w.squeeze(2)
    asp_c1_b = tensors["speaker_encoder.asp.tdnn.conv.bias"]
    asp_h = np.tanh(relu(conv1d_k1(asp_input, asp_c1_w, asp_c1_b)))  # [128, T]

    # conv2: [128->1536, k=1]
    asp_c2_w = tensors["speaker_encoder.asp.conv.weight"]
    if asp_c2_w.ndim == 3:
        asp_c2_w = asp_c2_w.squeeze(2)
    asp_c2_b = tensors["speaker_encoder.asp.conv.bias"]
    attn = conv1d_k1(asp_h, asp_c2_w, asp_c2_b)  # [1536, T]

    # Softmax over T
    attn_max = attn.max(axis=1, keepdims=True)
    attn_exp = np.exp(attn - attn_max)
    attn = attn_exp / attn_exp.sum(axis=1, keepdims=True)

    # Weighted mean and std
    w_mean = (attn * hidden).sum(axis=1)  # [1536]
    w_sq = (attn * hidden * hidden).sum(axis=1)
    w_var = w_sq - w_mean * w_mean
    w_var = np.maximum(w_var, 1e-10)
    w_std = np.sqrt(w_var)

    # FC: concat(mean, std) -> [3072] -> [1024]
    fc_input = np.concatenate([w_mean, w_std])  # [3072]
    fc_w = tensors["speaker_encoder.fc.weight"]
    if fc_w.ndim == 3:
        fc_w = fc_w.squeeze(2)
    fc_b = tensors["speaker_encoder.fc.bias"]
    embedding = fc_w @ fc_input + fc_b  # [1024]

    return embedding


# ---- Test ----

def generate_deterministic_mel(n_frames=100, seed=42):
    """Generate a deterministic mel spectrogram for testing."""
    rng = np.random.RandomState(seed)
    # Simulate realistic mel values (log-scale, mostly negative)
    mel = rng.randn(128, n_frames).astype(np.float32) * 2.0 - 4.0
    return mel


def main():
    # Find model
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        model_path = sys.argv[1]
    else:
        model_path = find_model_path()

    if not model_path or not os.path.exists(model_path):
        print("ERROR: model.safetensors not found.")
        print("Usage: python tools/verify_speaker_enc.py [path/to/model.safetensors]")
        print("Or set DEPS_ROOT environment variable.")
        sys.exit(1)

    print(f"Loading {model_path}...")
    tensors = load_safetensors(model_path)

    # Check for speaker encoder tensors
    spk_tensors = {k: v for k, v in tensors.items() if k.startswith("speaker_encoder.")}
    if not spk_tensors:
        print("ERROR: No speaker_encoder.* tensors found (not a Base model?)")
        sys.exit(1)
    print(f"  Found {len(spk_tensors)} speaker encoder tensors")

    # Print weight shapes
    key_tensors = [
        "speaker_encoder.blocks.0.conv.weight",
        "speaker_encoder.blocks.1.tdnn1.conv.weight",
        "speaker_encoder.blocks.1.res2net_block.blocks.0.conv.weight",
        "speaker_encoder.mfa.conv.weight",
        "speaker_encoder.asp.tdnn.conv.weight",
        "speaker_encoder.asp.conv.weight",
        "speaker_encoder.fc.weight",
    ]
    print("\n  Key tensor shapes:")
    for name in key_tensors:
        if name in tensors:
            print(f"    {name}: {tensors[name].shape}")

    # Generate deterministic mel input
    n_frames = 100
    print(f"\nGenerating deterministic mel input [{128}, {n_frames}] (seed=42)...")
    mel = generate_deterministic_mel(n_frames, seed=42)
    print(f"  range=[{mel.min():.3f}, {mel.max():.3f}]")

    # Run forward pass
    print("\nRunning ECAPA-TDNN forward pass...")
    embedding = ecapa_tdnn_forward(tensors, mel, n_frames)
    print(f"  embedding shape: {embedding.shape}")
    print(f"  embedding range: [{embedding.min():.6f}, {embedding.max():.6f}]")
    print(f"  embedding norm: {np.linalg.norm(embedding):.6f}")
    print(f"  embedding[:8]: {embedding[:8]}")

    # Sanity checks
    all_ok = True

    # 1. Output dimension
    if embedding.shape != (1024,):
        print(f"  FAIL: expected shape (1024,), got {embedding.shape}")
        all_ok = False
    else:
        print("  Dimension check: PASS (1024)")

    # 2. Non-trivial output (not all zeros or NaN)
    if np.all(embedding == 0):
        print("  FAIL: all-zero output")
        all_ok = False
    elif np.any(np.isnan(embedding)):
        print("  FAIL: NaN in output")
        all_ok = False
    elif np.any(np.isinf(embedding)):
        print("  FAIL: Inf in output")
        all_ok = False
    else:
        print("  Non-trivial check: PASS")

    # 3. Determinism: run again with same input
    embedding2 = ecapa_tdnn_forward(tensors, mel, n_frames)
    det_diff = np.abs(embedding - embedding2).max()
    if det_diff > 0:
        print(f"  FAIL: non-deterministic (max_diff={det_diff:.8f})")
        all_ok = False
    else:
        print("  Determinism check: PASS (identical on re-run)")

    # 4. Different input -> different output
    mel2 = generate_deterministic_mel(n_frames, seed=99)
    embedding3 = ecapa_tdnn_forward(tensors, mel2, n_frames)
    diff_input_corr = np.corrcoef(embedding.flatten(), embedding3.flatten())[0, 1]
    if diff_input_corr > 0.999:
        print(f"  FAIL: different inputs produce too-similar outputs (corr={diff_input_corr:.6f})")
        all_ok = False
    else:
        print(f"  Sensitivity check: PASS (different input -> corr={diff_input_corr:.4f})")

    # 5. Compare against C dump if available
    c_dump = "spkenc_out.raw"
    if os.path.exists(c_dump):
        print(f"\n--- C dump comparison ({c_dump}) ---")
        c_emb = np.fromfile(c_dump, dtype=np.float32)
        if c_emb.shape != (1024,):
            print(f"  FAIL: C dump shape {c_emb.shape}, expected (1024,)")
        else:
            diff = np.abs(embedding - c_emb)
            max_diff = diff.max()
            corr = np.corrcoef(embedding, c_emb)[0, 1]
            ok = corr > 0.999
            status = "PASS" if ok else "FAIL"
            print(f"  Python vs C: {status} (corr={corr:.6f}, max_diff={max_diff:.6f})")
            if not ok:
                all_ok = False

    # Save Python output for comparison
    embedding.tofile("py_spkenc_out.raw")
    print(f"\n  Saved Python embedding to py_spkenc_out.raw")

    if all_ok:
        print("\nAll checks PASSED")
    else:
        print("\nSome checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
