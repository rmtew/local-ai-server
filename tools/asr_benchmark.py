#!/usr/bin/env python3
"""ASR benchmark tool for local-ai-server.

Runs transcription requests against the server and collects timing data.
Produces a summary table with encode/decode breakdowns, RTF, and throughput.

Usage:
    python tools/asr_benchmark.py                    # default: 3 iterations, all samples
    python tools/asr_benchmark.py --iterations 5     # more iterations
    python tools/asr_benchmark.py --url http://host:port  # custom server
    python tools/asr_benchmark.py --samples jfk.wav  # specific file(s)
    python tools/asr_benchmark.py --save              # append results to tools/asr_benchmarks.md
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SAMPLES_DIR = PROJECT_DIR / "qwen-asr" / "samples"
BENCHMARKS_FILE = SCRIPT_DIR / "asr_benchmarks.md"

DEFAULT_URL = "http://localhost:8090"
DEFAULT_ITERATIONS = 3


def find_samples(names=None):
    """Find WAV files to benchmark."""
    if names:
        files = []
        for name in names:
            p = SAMPLES_DIR / name
            if not p.exists():
                p = Path(name)
            if p.exists():
                files.append(p)
            else:
                print(f"Warning: sample not found: {name}", file=sys.stderr)
        return files

    # Default: all .wav files in samples dir, sorted by size
    files = sorted(SAMPLES_DIR.glob("*.wav"), key=lambda f: f.stat().st_size)
    return files


def wav_duration_sec(path):
    """Read duration from WAV header (assumes 16-bit mono 16kHz)."""
    size = path.stat().st_size
    return max(0, (size - 44)) / 32000


def check_server(url):
    """Check if server is healthy."""
    try:
        r = subprocess.run(
            ["curl", "-s", "--max-time", "5", f"{url}/health"],
            capture_output=True, text=True
        )
        data = json.loads(r.stdout)
        return data.get("status") == "ok"
    except Exception:
        return False


def transcribe(url, wav_path):
    """Send transcription request via curl and return parsed response with wall time."""
    wall_start = time.perf_counter()
    r = subprocess.run(
        [
            "curl", "-s", "-X", "POST",
            f"{url}/v1/audio/transcriptions",
            "-F", f"file=@{wav_path}",
            "-F", "response_format=verbose_json",
        ],
        capture_output=True, text=True, timeout=120,
    )
    wall_ms = (time.perf_counter() - wall_start) * 1000

    if r.returncode != 0:
        raise RuntimeError(f"curl failed: {r.stderr}")

    result = json.loads(r.stdout)
    result["wall_ms"] = wall_ms
    return result


def run_benchmark(url, samples, iterations, warmup=1):
    """Run benchmark across all samples."""
    results = {}

    for sample in samples:
        audio_sec = wav_duration_sec(sample)
        name = sample.name
        runs = []

        # Warmup
        for _ in range(warmup):
            try:
                transcribe(url, sample)
            except Exception as e:
                print(f"  Warmup failed for {name}: {e}", file=sys.stderr)
                break

        # Timed runs
        for i in range(iterations):
            try:
                r = transcribe(url, sample)
                run_data = {
                    "total_ms": r["perf_total_ms"],
                    "encode_ms": r["perf_encode_ms"],
                    "decode_ms": r["perf_decode_ms"],
                    "wall_ms": r["wall_ms"],
                    "audio_sec": r["duration"],
                    "words": len(r.get("words", [])),
                    "text": r["text"],
                }
                runs.append(run_data)
            except Exception as e:
                print(f"  Run {i+1} failed for {name}: {e}", file=sys.stderr)

        if runs:
            results[name] = {
                "audio_sec": audio_sec,
                "runs": runs,
            }

    return results


def median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def format_results(results, iterations):
    """Format results as a text table."""
    lines = []

    lines.append(f"{'Sample':<22} {'Audio':>6} {'Total':>7} {'Encode':>7} {'Decode':>7} "
                 f"{'Wall':>7} {'RTF':>6} {'Words':>5}  Text")
    lines.append("-" * 100)

    for name, data in results.items():
        runs = data["runs"]
        audio = data["audio_sec"]

        total = median([r["total_ms"] for r in runs])
        encode = median([r["encode_ms"] for r in runs])
        decode = median([r["decode_ms"] for r in runs])
        wall = median([r["wall_ms"] for r in runs])
        words = runs[0]["words"]
        text = runs[0]["text"][:50]
        rtf = total / (audio * 1000) if audio > 0 else 0

        for i, r in enumerate(runs):
            tag = f"  run {i+1}"
            r_rtf = r["total_ms"] / (audio * 1000) if audio > 0 else 0
            lines.append(f"{tag:<22} {audio:>5.1f}s {r['total_ms']:>6.0f}ms {r['encode_ms']:>6.0f}ms "
                         f"{r['decode_ms']:>6.0f}ms {r['wall_ms']:>6.0f}ms {r_rtf:>5.2f}x {r['words']:>5}")

        lines.append(f"  {'>>> MEDIAN':<20} {audio:>5.1f}s {total:>6.0f}ms {encode:>6.0f}ms "
                      f"{decode:>6.0f}ms {wall:>6.0f}ms {rtf:>5.2f}x {words:>5}  \"{text}\"")
        lines.append("")

    lines.append("Metrics: Total/Encode/Decode = server-side perf counters, Wall = curl round-trip")
    lines.append("RTF = Real-Time Factor (total_ms / audio_ms, lower is faster)")

    return "\n".join(lines)


def format_markdown(results, iterations, notes=""):
    """Format results as markdown for the benchmarks log."""
    lines = []
    ts = time.strftime("%Y-%m-%d %H:%M")
    lines.append(f"### {ts}")
    if notes:
        lines.append(f"\n{notes}\n")

    lines.append("")
    lines.append("| Sample | Audio | Total (ms) | Encode (ms) | Decode (ms) | Wall (ms) | RTF | Words |")
    lines.append("|--------|------:|-----------:|------------:|------------:|----------:|----:|------:|")

    for name, data in results.items():
        runs = data["runs"]
        audio = data["audio_sec"]
        total = median([r["total_ms"] for r in runs])
        encode = median([r["encode_ms"] for r in runs])
        decode = median([r["decode_ms"] for r in runs])
        wall = median([r["wall_ms"] for r in runs])
        words = runs[0]["words"]
        rtf = total / (audio * 1000) if audio > 0 else 0

        lines.append(f"| {name} | {audio:.1f}s | {total:.0f} | {encode:.0f} | {decode:.0f} | {wall:.0f} | {rtf:.2f}x | {words} |")

    lines.append("")
    return "\n".join(lines)


def get_server_info(url):
    """Try to get model info from /v1/models."""
    try:
        r = subprocess.run(
            ["curl", "-s", "--max-time", "5", f"{url}/v1/models"],
            capture_output=True, text=True
        )
        data = json.loads(r.stdout)
        models = data.get("data", [])
        if models:
            return models[0].get("id", "unknown")
    except Exception:
        pass
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="ASR benchmark tool")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Server URL (default: {DEFAULT_URL})")
    parser.add_argument("--iterations", "-n", type=int, default=DEFAULT_ITERATIONS, help="Iterations per sample")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (default: 1)")
    parser.add_argument("--samples", nargs="+", help="Specific WAV file(s) to test")
    parser.add_argument("--save", action="store_true", help="Append results to asr_benchmarks.md")
    parser.add_argument("--notes", default="", help="Notes to include in saved results")
    args = parser.parse_args()

    if not check_server(args.url):
        print(f"Error: Server not responding at {args.url}/health", file=sys.stderr)
        print("Start the server first:", file=sys.stderr)
        print("  bin/local-ai-server.exe --model=<path> --port=8090 --verbose", file=sys.stderr)
        sys.exit(1)

    model_id = get_server_info(args.url)
    print(f"Server: {args.url}  Model: {model_id}")

    samples = find_samples(args.samples)
    if not samples:
        print("Error: No WAV samples found", file=sys.stderr)
        sys.exit(1)

    print(f"Samples: {len(samples)}  Iterations: {args.iterations}  Warmup: {args.warmup}")
    print(f"Files: {', '.join(s.name for s in samples)}")
    print()

    results = run_benchmark(args.url, samples, args.iterations, args.warmup)

    if not results:
        print("Error: No successful runs", file=sys.stderr)
        sys.exit(1)

    print(format_results(results, args.iterations))

    if args.save:
        notes = args.notes or f"Model: {model_id}, {args.iterations} iterations, {args.warmup} warmup"
        md = format_markdown(results, args.iterations, notes)

        if BENCHMARKS_FILE.exists():
            existing = BENCHMARKS_FILE.read_text()
        else:
            existing = "# ASR Benchmarks\n\nPerformance tracking for ASR inference.\n\n"
            existing += "Hardware: See individual entries for details.\n\n---\n"

        with open(BENCHMARKS_FILE, "w") as f:
            f.write(existing + "\n" + md + "\n")

        print(f"\nResults saved to {BENCHMARKS_FILE}")


if __name__ == "__main__":
    main()
