"""Generate voice_presets.bin from reference WAV files.

Loads the TTS model's speaker encoder weights, computes mel spectrograms
from reference WAV files, runs the ECAPA-TDNN forward pass, and writes
precomputed 1024-dim speaker embeddings in the binary format expected
by tts_voice_presets.c.

Binary format:
  [n:int32] count of presets
  n x [name:64 bytes (null-padded), embed:1024 x float32]

Usage:
    python tools/generate_voice_presets.py voice_samples/
    python tools/generate_voice_presets.py voice_samples/ --output presets.bin
    python tools/generate_voice_presets.py voice_samples/ --list

WAV files must be 24kHz mono 16-bit PCM. The preset name is derived from
the filename (e.g. alloy.wav -> "alloy").
"""
import numpy as np
import struct
import json
import sys
import os
import argparse

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# Import mel and speaker encoder from sibling scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from verify_mel import mel_spectrogram_A as compute_mel, load_wav
from verify_speaker_enc import (
    load_safetensors, find_model_path, ecapa_tdnn_forward
)

PRESET_NAME_LEN = 64
PRESET_EMBED_DIM = 1024


def write_presets_bin(presets, output_path):
    """Write presets to binary file.
    presets: list of (name, embedding) tuples."""
    with open(output_path, "wb") as f:
        f.write(struct.pack("<i", len(presets)))
        for name, embed in presets:
            # Name: 64 bytes, null-padded
            name_bytes = name.encode("utf-8")[:PRESET_NAME_LEN - 1]
            name_padded = name_bytes + b"\x00" * (PRESET_NAME_LEN - len(name_bytes))
            f.write(name_padded)
            # Embedding: 1024 x float32
            f.write(embed.astype(np.float32).tobytes())
    print(f"Wrote {len(presets)} presets to {output_path}")
    print(f"  File size: {os.path.getsize(output_path):,} bytes")


def read_presets_bin(path):
    """Read and display presets from binary file."""
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        presets = []
        for _ in range(n):
            name = f.read(PRESET_NAME_LEN).split(b"\x00")[0].decode("utf-8")
            embed = np.frombuffer(f.read(PRESET_EMBED_DIM * 4), dtype=np.float32).copy()
            presets.append((name, embed))
    return presets


def main():
    parser = argparse.ArgumentParser(description="Generate voice presets")
    parser.add_argument("wav_dir", help="Directory containing reference WAV files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (default: <tts-model-dir>/voice_presets.bin)")
    parser.add_argument("--model", "-m", default=None,
                        help="Path to model.safetensors")
    parser.add_argument("--list", action="store_true",
                        help="List presets in existing binary file instead of generating")
    args = parser.parse_args()

    # List mode
    if args.list:
        bin_path = args.output or os.path.join(args.wav_dir, "voice_presets.bin")
        if not os.path.exists(bin_path):
            print(f"File not found: {bin_path}")
            sys.exit(1)
        presets = read_presets_bin(bin_path)
        print(f"{len(presets)} presets in {bin_path}:")
        for name, embed in presets:
            print(f"  {name:20s}  norm={np.linalg.norm(embed):.4f}  "
                  f"range=[{embed.min():.4f}, {embed.max():.4f}]")
        return

    # Find WAV files
    wav_dir = args.wav_dir
    if not os.path.isdir(wav_dir):
        print(f"ERROR: {wav_dir} is not a directory")
        sys.exit(1)

    wav_files = sorted([f for f in os.listdir(wav_dir)
                        if f.lower().endswith(".wav")])
    if not wav_files:
        print(f"No .wav files found in {wav_dir}")
        sys.exit(1)

    print(f"Found {len(wav_files)} WAV files in {wav_dir}")

    # Load model
    model_path = args.model or find_model_path()
    if not model_path or not os.path.exists(model_path):
        print("ERROR: model.safetensors not found")
        print("Set DEPS_ROOT or use --model flag")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    tensors = load_safetensors(model_path)
    spk_count = sum(1 for k in tensors if k.startswith("speaker_encoder."))
    if spk_count == 0:
        print("ERROR: No speaker_encoder.* tensors (not a Base model)")
        sys.exit(1)
    print(f"  Loaded {spk_count} speaker encoder tensors")

    # Process each WAV
    presets = []
    for wav_file in wav_files:
        name = os.path.splitext(wav_file)[0]
        if len(name) >= PRESET_NAME_LEN:
            print(f"  WARNING: name '{name}' too long, truncating to {PRESET_NAME_LEN - 1} chars")
            name = name[:PRESET_NAME_LEN - 1]

        wav_path = os.path.join(wav_dir, wav_file)
        print(f"\n  Processing {wav_file} -> '{name}'...")
        audio = load_wav(wav_path)

        # Compute mel
        mel, n_frames = compute_mel(audio)
        print(f"    mel: [{mel.shape[0]}, {mel.shape[1]}]")

        # Run speaker encoder
        embedding = ecapa_tdnn_forward(tensors, mel, n_frames)
        print(f"    embedding: norm={np.linalg.norm(embedding):.4f}, "
              f"range=[{embedding.min():.4f}, {embedding.max():.4f}]")

        presets.append((name, embedding))

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        model_dir = os.path.dirname(model_path)
        output_path = os.path.join(model_dir, "voice_presets.bin")

    # Write binary file
    print()
    write_presets_bin(presets, output_path)

    # Verify by reading back
    print("\nVerification (read-back):")
    readback = read_presets_bin(output_path)
    all_ok = True
    for (orig_name, orig_embed), (rb_name, rb_embed) in zip(presets, readback):
        match = (orig_name == rb_name and np.allclose(orig_embed, rb_embed))
        status = "ok" if match else "MISMATCH"
        if not match:
            all_ok = False
        print(f"  {orig_name}: {status}")

    if all_ok:
        print("\nAll presets verified OK")
    else:
        print("\nVerification FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
