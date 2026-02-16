#!/usr/bin/env python3
"""
TTS HTTP benchmark — measures wall-clock time via curl to compare with
native in-process benchmark (tts-bench.exe).

Same texts, same seed, same parameters. Difference = HTTP overhead.

Usage:
    python tools/tts_bench_http.py [--runs N] [--warmup N] [--seed N]
"""

import json
import struct
import sys
import time
import urllib.request
import urllib.error

SERVER = "http://127.0.0.1:8090"
SEED = 42
RUNS = 3
WARMUP = 1

CASES = [
    ("4s",
     "The quick brown fox jumps over the lazy dog near the riverbank."),
    ("6s",
     "She sells seashells by the seashore, and the shells she sells "
     "are seashells, I'm sure."),
    ("8s",
     "The old lighthouse keeper climbed the spiral staircase, his weathered "
     "hands gripping the iron railing as wind howled through the cracks "
     "in the ancient stone walls."),
    ("10s",
     "In the beginning, the universe was created. This has made a lot of "
     "people very angry and been widely regarded as a bad move. Many were "
     "increasingly of the opinion that they had all made a big mistake "
     "in coming down from the trees in the first place."),
    ("12s",
     "It was a bright cold day in April, and the clocks were striking thirteen. "
     "Winston Smith, his chin nuzzled into his breast in an effort to escape "
     "the vile wind, slipped quickly through the glass doors of Victory Mansions, "
     "though not quickly enough to prevent a swirl of gritty dust from entering "
     "along with him."),
    ("16s",
     "It was a bright cold day in April, and the clocks were striking thirteen. "
     "Winston Smith, his chin nuzzled into his breast in an effort to escape "
     "the vile wind, slipped quickly through the glass doors of Victory Mansions, "
     "though not quickly enough to prevent a swirl of gritty dust from entering "
     "along with him. The hallway smelt of boiled cabbage and old rag mats. "
     "At one end of it a coloured poster, too large for indoor display, had been "
     "tacked to the wall."),
]


def wav_duration(data):
    """Get WAV duration from raw bytes."""
    if len(data) < 44 or data[:4] != b"RIFF":
        return 0.0
    # Find data chunk
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos+4]
        chunk_size = struct.unpack_from("<I", data, pos+4)[0]
        if chunk_id == b"fmt ":
            sample_rate = struct.unpack_from("<I", data, pos+12)[0]
            bits = struct.unpack_from("<H", data, pos+22)[0]
            channels = struct.unpack_from("<H", data, pos+10)[0]
        elif chunk_id == b"data":
            n_samples = chunk_size // (bits // 8 * channels)
            return n_samples / sample_rate
        pos += 8 + chunk_size
    return 0.0


def tts_request(text, seed):
    """Send TTS request, return (wav_bytes, wall_ms)."""
    payload = json.dumps({
        "input": text,
        "voice": "alloy",
        "seed": seed,
        "temperature": 0.3,
        "top_k": 50,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    wall_ms = (time.perf_counter() - t0) * 1000
    return data, wall_ms


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=RUNS)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    # Check server
    try:
        urllib.request.urlopen(f"{SERVER}/health", timeout=5)
    except Exception:
        print(f"Error: server not responding at {SERVER}/health")
        sys.exit(1)

    print(f"=== TTS HTTP Benchmark ===")
    print(f"Server:  {SERVER}")
    print(f"Runs:    {args.runs} (+ {args.warmup} warmup)")
    print(f"Seed:    {args.seed}")
    print()

    # Warmup
    if args.warmup > 0:
        print(f"Warmup ({args.warmup} run(s))...")
        for w in range(args.warmup):
            data, ms = tts_request(CASES[0][1], args.seed)
            dur = wav_duration(data)
            print(f"  warmup {w+1}: {dur:.1f}s audio, {ms:.0f} ms")
        print()

    # Benchmark
    results = []
    for case_id, text in CASES:
        print(f"  {case_id:<12} \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        runs_data = []
        for r in range(args.runs):
            data, wall_ms = tts_request(text, args.seed)
            dur = wav_duration(data)
            n_samples = int(dur * 24000)
            # Estimate steps from duration (each step = 80ms = 1920 samples)
            steps = n_samples // 1920
            rtf = wall_ms / (dur * 1000) if dur > 0 else 0
            runs_data.append({
                "wall_ms": wall_ms,
                "audio_sec": dur,
                "steps": steps,
                "rtf": rtf,
                "wav_size": len(data),
            })
            print(f"    run {r+1}: {wall_ms:6.0f} ms | ~{steps:3d} steps | "
                  f"{dur:5.1f}s audio | RTF {rtf:.3f}")

        results.append({
            "id": case_id,
            "text_len": len(text),
            "runs": runs_data,
            "median_wall": median([r["wall_ms"] for r in runs_data]),
            "median_rtf": median([r["rtf"] for r in runs_data]),
            "audio_sec": runs_data[0]["audio_sec"],
            "steps": runs_data[0]["steps"],
        })
        print()

    # Summary
    print(f"=== Summary (median of {args.runs} runs, seed={args.seed}) ===")
    print()
    print(f"{'Case':<12} {'Steps':>5} {'Audio':>6} {'Wall':>8} {'RTF':>7} {'chars':>5}")
    print(f"{'------------':<12} {'-----':>5} {'------':>6} {'--------':>8} {'-------':>7} {'-----':>5}")

    for r in results:
        print(f"{r['id']:<12} {r['steps']:>5} {r['audio_sec']:>5.1f}s "
              f"{r['median_wall']:>6.0f} ms {r['median_rtf']:>5.3f}x {r['text_len']:>5}")

    print()
    print("Compare with tts-bench.exe native results to see HTTP overhead.")


if __name__ == "__main__":
    main()
