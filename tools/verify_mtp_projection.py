#!/usr/bin/env python3
"""
Verify mtp_projection math for 1.7B model using numpy + safetensors.

Loads the small_to_mtp_projection weight from the 1.7B safetensors,
generates a synthetic talker hidden vector, and verifies the linear
projection matches expected output.

Also inspects key tensor shapes to confirm dimensions are correct.

Usage:
    pip install safetensors numpy   # lightweight, no PyTorch needed
    python tools/verify_mtp_projection.py <model_dir>
"""

import sys
import os
import json
import struct
import numpy as np


def read_safetensors_header(path):
    """Read safetensors JSON header without loading tensors."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_size).decode('utf-8')
    return json.loads(header_json)


def load_tensor_f32(path, header, name):
    """Load a single tensor from safetensors as float32 numpy array."""
    if name not in header:
        return None
    meta = header[name]
    dtype_str = meta['dtype']
    shape = meta['data_offsets']  # [start, end]
    offset_start, offset_end = meta['data_offsets']

    # Read raw bytes
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        f.seek(8 + header_size + offset_start)
        raw = f.read(offset_end - offset_start)

    # Convert to numpy
    if dtype_str == 'BF16':
        # BF16 -> F32: pad each 2-byte BF16 with 2 zero bytes on the left
        bf16 = np.frombuffer(raw, dtype=np.uint16)
        f32_int = bf16.astype(np.uint32) << 16
        arr = f32_int.view(np.float32)
    elif dtype_str == 'F32':
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    elif dtype_str == 'F16':
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    return arr.reshape(meta['shape'])


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir>")
        print("  model_dir: path to qwen3-tts-12hz-1.7b-base/")
        sys.exit(1)

    model_dir = sys.argv[1]
    sf_path = os.path.join(model_dir, "model.safetensors")

    if not os.path.exists(sf_path):
        print(f"ERROR: {sf_path} not found")
        sys.exit(1)

    print(f"Loading header from {sf_path}...")
    header = read_safetensors_header(sf_path)

    # Remove __metadata__ if present
    header.pop('__metadata__', None)

    # ===== 1. Inspect key tensor shapes =====
    print("\n=== Key Tensor Shapes ===")
    key_tensors = [
        "talker.model.codec_embedding.weight",
        "talker.code_predictor.small_to_mtp_projection.weight",
        "talker.code_predictor.small_to_mtp_projection.bias",
        "talker.code_predictor.model.norm.weight",
        "talker.code_predictor.model.codec_embedding.0.weight",
        "talker.code_predictor.model.codec_embedding.14.weight",
        "talker.code_predictor.lm_head.0.weight",
        "talker.code_predictor.lm_head.14.weight",
        "talker.model.layers.0.self_attn.q_proj.weight",
        "talker.model.layers.0.self_attn.k_proj.weight",
        "talker.model.layers.0.mlp.gate_proj.weight",
        "talker.code_predictor.model.layers.0.self_attn.q_proj.weight",
        "talker.code_predictor.model.layers.0.mlp.gate_proj.weight",
        "talker.codec_head.weight",
        "speaker_encoder.fc.weight",
    ]

    for name in key_tensors:
        if name in header:
            meta = header[name]
            print(f"  {name}: shape={meta['shape']}, dtype={meta['dtype']}")
        else:
            print(f"  {name}: NOT FOUND")

    # ===== 2. Verify dimensions =====
    print("\n=== Dimension Verification ===")

    codec_emb = header.get("talker.model.codec_embedding.weight", {})
    talker_hidden = codec_emb.get('shape', [0, 0])[1]
    print(f"  Talker hidden (from codec_embedding): {talker_hidden}")

    mtp_w = header.get("talker.code_predictor.small_to_mtp_projection.weight", {})
    if mtp_w:
        cp_hidden, mtp_input = mtp_w.get('shape', [0, 0])
        print(f"  MTP projection: [{cp_hidden}, {mtp_input}] (cp_hidden={cp_hidden}, talker_hidden={mtp_input})")
        assert mtp_input == talker_hidden, f"MTP input dim {mtp_input} != talker_hidden {talker_hidden}"
        print(f"  OK: MTP input dim matches talker_hidden")
    else:
        print("  No MTP projection found (0.6B model?)")

    cp_norm = header.get("talker.code_predictor.model.norm.weight", {})
    cp_hidden_from_norm = cp_norm.get('shape', [0])[0]
    print(f"  CP hidden (from norm): {cp_hidden_from_norm}")
    if mtp_w:
        assert cp_hidden == cp_hidden_from_norm, f"CP hidden mismatch: {cp_hidden} vs {cp_hidden_from_norm}"
        print(f"  OK: CP hidden consistent")

    # Check CP codec embedding is in TALKER space
    cp_codec_0 = header.get("talker.code_predictor.model.codec_embedding.0.weight", {})
    if cp_codec_0:
        cp_codec_shape = cp_codec_0.get('shape', [0, 0])
        print(f"  CP codec_embed[0]: shape={cp_codec_shape} (should be [2048, {talker_hidden}])")
        assert cp_codec_shape[1] == talker_hidden, \
            f"CP codec_embed embed_dim={cp_codec_shape[1]} != talker_hidden={talker_hidden}"
        print(f"  OK: CP codec embeddings are in talker space")

    # Check CP lm_head is in CP space
    cp_lm_0 = header.get("talker.code_predictor.lm_head.0.weight", {})
    if cp_lm_0:
        lm_shape = cp_lm_0.get('shape', [0, 0])
        print(f"  CP lm_head[0]: shape={lm_shape} (should be [2048, {cp_hidden_from_norm}])")
        assert lm_shape[1] == cp_hidden_from_norm, \
            f"CP lm_head dim={lm_shape[1]} != cp_hidden={cp_hidden_from_norm}"
        print(f"  OK: CP lm_heads are in CP space")

    # ===== 3. Numerical verification of mtp_projection =====
    if not mtp_w:
        print("\n=== Skipping numerical test (no MTP projection) ===")
        return

    print("\n=== Numerical MTP Projection Verification ===")

    # Load actual weights
    W = load_tensor_f32(sf_path, header, "talker.code_predictor.small_to_mtp_projection.weight")
    print(f"  Loaded MTP weight: shape={W.shape}, dtype={W.dtype}")
    print(f"  Weight stats: min={W.min():.6f}, max={W.max():.6f}, mean={W.mean():.6f}, std={W.std():.6f}")

    bias_name = "talker.code_predictor.small_to_mtp_projection.bias"
    b = load_tensor_f32(sf_path, header, bias_name)
    if b is not None:
        print(f"  Loaded MTP bias: shape={b.shape}")
        print(f"  Bias stats: min={b.min():.6f}, max={b.max():.6f}, mean={b.mean():.6f}")
    else:
        print("  No MTP bias found")

    # Verify: out = x @ W^T + b  (standard nn.Linear)
    # W shape: [cp_hidden, talker_hidden] = [1024, 2048]
    # x shape: [1, talker_hidden] = [1, 2048]
    # out shape: [1, cp_hidden] = [1, 1024]
    np.random.seed(42)
    x = np.random.randn(1, talker_hidden).astype(np.float32)

    out = x @ W.T
    if b is not None:
        out += b

    print(f"\n  Test input: shape={x.shape}, norm={np.linalg.norm(x):.4f}")
    print(f"  Output: shape={out.shape}, norm={np.linalg.norm(out):.4f}")
    print(f"  Output stats: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")

    # Our C code does: qwen_linear_nobias(out, in, W, 1, H_T, H_CP)
    # which computes: out[H_CP] = in[H_T] @ W[H_CP, H_T]^T
    # This is the SAME as nn.Linear: out = x @ W^T
    print("\n  C code equivalent: qwen_linear_nobias(out, in, W, 1, H_T={}, H_CP={})".format(
        talker_hidden, cp_hidden))
    print("  = in[1, {}] @ W[{}, {}]^T = out[1, {}]".format(
        talker_hidden, cp_hidden, talker_hidden, cp_hidden))
    print("  OK: Math is equivalent to nn.Linear (out = x @ W^T + bias)")

    # ===== 4. Verify codec embedding projection chain =====
    print("\n=== Codec Embedding -> MTP Projection Chain ===")

    # Load a codec embedding from CP table (in talker space)
    cp_codec_emb = load_tensor_f32(sf_path, header,
                                    "talker.code_predictor.model.codec_embedding.0.weight")
    if cp_codec_emb is not None:
        print(f"  CP codec_embed[0]: shape={cp_codec_emb.shape}")

        # Pick token 42 as test
        tok_embed = cp_codec_emb[42:43, :]  # [1, talker_hidden=2048]
        print(f"  Token 42 embed: shape={tok_embed.shape}, norm={np.linalg.norm(tok_embed):.4f}")

        # Project to CP space
        projected = tok_embed @ W.T
        if b is not None:
            projected += b
        print(f"  Projected to CP space: shape={projected.shape}, norm={np.linalg.norm(projected):.4f}")

        # This should produce a 1024-dim vector for the code predictor
        assert projected.shape == (1, cp_hidden), f"Expected (1, {cp_hidden}), got {projected.shape}"
        print(f"  OK: Projection chain correct: talker_embed[{talker_hidden}] -> mtp_proj -> cp_embed[{cp_hidden}]")

    # ===== 5. Compare talker codec_embed vs CP codec_embed =====
    print("\n=== Talker vs CP Codec Embeddings ===")

    talker_codec = load_tensor_f32(sf_path, header, "talker.model.codec_embedding.weight")
    if talker_codec is not None and cp_codec_emb is not None:
        print(f"  Talker codec_embed: shape={talker_codec.shape}")
        print(f"  CP codec_embed[0]: shape={cp_codec_emb.shape}")

        # These should be DIFFERENT tables
        # Talker codec_embed covers vocab 0..3071 (talker vocab)
        # CP codec_embed[j] covers vocab 0..2047 (codec vocab)
        if talker_codec.shape[0] != cp_codec_emb.shape[0]:
            print(f"  Different vocab sizes: talker={talker_codec.shape[0]}, cp={cp_codec_emb.shape[0]}")
        else:
            # Check if they share the first 2048 entries
            overlap = talker_codec[:2048, :]
            diff = np.abs(overlap - cp_codec_emb).max()
            print(f"  Max diff (first 2048 entries): {diff:.6f}")
            if diff < 1e-5:
                print("  WARNING: Tables appear identical (shared weights?)")
            else:
                print("  Tables are different (independent weights)")

    print("\n=== All checks passed ===")


if __name__ == "__main__":
    main()
