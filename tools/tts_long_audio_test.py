#!/usr/bin/env python3
"""
TTS Long Audio Quality Test

Generates audio at increasing lengths and checks for degeneration:
- CB0 repetition rate over time (sliding window)
- Per-segment entropy collapse
- Audio amplitude analysis (silence, clipping, RMS over time)
- Natural EOS vs forced truncation

Requires: running local-ai-server with TTS_DUMP_CODES env var set.
Usage:
    # Start server (script will do this if --start-server is passed):
    TTS_DUMP_CODES=tts_long_test/codes.txt bin/local-ai-server.exe \
        --tts-model=<model-dir> --fp16 --verbose --port=8090

    python tts_long_audio_test.py [--server-url=http://localhost:8090] [--seed=42]
    python tts_long_audio_test.py --start-server=<model-dir> [--fp16] [--seed=42]
"""

import os
import sys
import time
import struct
import subprocess
import json
import argparse
import numpy as np
from collections import Counter
from urllib.request import Request, urlopen
from urllib.error import URLError


# ---- Test texts of increasing length ----
TEST_CASES = [
    ("short", "Hello world, this is a test."),
    ("medium",
     "The quick brown fox jumps over the lazy dog. "
     "She sells seashells by the seashore on sunny summer days."),
    ("long",
     "The old lighthouse keeper climbed the spiral staircase, his weathered "
     "hands gripping the iron railing as wind howled through the cracks in "
     "the ancient stone walls. At the top, the great lens waited, ready to "
     "cast its beam across the dark and stormy waters below."),
    ("very_long",
     "In the beginning, the universe was created. This has made a lot of "
     "people very angry and been widely regarded as a bad move. Many were "
     "increasingly of the opinion that they had all made a big mistake in "
     "coming down from the trees in the first place. And some said that even "
     "the trees had been a bad move, and that no one should ever have left "
     "the oceans. Meanwhile, the poor Babel fish, by effectively removing "
     "all barriers to communication between different races and cultures, "
     "has caused more and bloodier wars than anything else in the history "
     "of creation."),
    ("extra_long",
     "It was a bright cold day in April, and the clocks were striking thirteen. "
     "Winston Smith, his chin nuzzled into his breast in an effort to escape "
     "the vile wind, slipped quickly through the glass doors of Victory Mansions, "
     "though not quickly enough to prevent a swirl of gritty dust from entering "
     "along with him. The hallway smelt of boiled cabbage and old rag mats. "
     "At one end of it a coloured poster, too large for indoor display, had been "
     "tacked to the wall. It depicted simply an enormous face, more than a metre "
     "wide: the face of a man of about forty-five, with a heavy black moustache "
     "and ruggedly handsome features. There was one of those pictures which are "
     "so contrived that the eyes follow you about when you move."),
]


def read_wav_samples(wav_path):
    """Read WAV file and return float samples + sample rate."""
    with open(wav_path, 'rb') as f:
        data = f.read()
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError("Not a valid WAV file")
    sample_rate = struct.unpack('<I', data[24:28])[0]
    bits = struct.unpack('<H', data[34:36])[0]
    data_start = data.index(b'data') + 8
    if bits == 16:
        pcm = np.frombuffer(data[data_start:], dtype=np.int16)
        samples = pcm.astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported bit depth: {bits}")
    return samples, sample_rate


def load_codes(path):
    """Load codec token dump: returns [n_steps, 16] int array or None."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = [int(x) for x in line.split('\t')]
            rows.append(tokens)
    if not rows:
        return None
    return np.array(rows, dtype=np.int64)


def analyze_codes(codes):
    """Analyze codec tokens for degeneration signals. Returns dict of metrics."""
    n_steps, n_groups = codes.shape
    cb0 = codes[:, 0]
    sub = codes[:, 1:]

    metrics = {
        'n_steps': n_steps,
        'cb0_unique': len(np.unique(cb0)),
        'cb0_mean': float(cb0.mean()),
        'cb0_std': float(cb0.std()),
    }

    # CB0 adjacent repetition rate
    repeats = sum(1 for i in range(1, len(cb0)) if cb0[i] == cb0[i - 1])
    metrics['cb0_repeat_rate'] = repeats / max(n_steps - 1, 1)

    # Sliding window repetition (window=10)
    window = 10
    window_repeats = []
    for start in range(0, n_steps - window + 1):
        w = cb0[start:start + window]
        wr = sum(1 for i in range(1, len(w)) if w[i] == w[i - 1])
        window_repeats.append(wr / (window - 1))
    metrics['cb0_max_window_repeat'] = max(window_repeats) if window_repeats else 0.0

    # Per-segment entropy (split into halves)
    def entropy(arr):
        counts = np.bincount(arr, minlength=2048)
        probs = counts[counts > 0] / counts.sum()
        return -np.sum(probs * np.log2(probs))

    half = n_steps // 2
    if half >= 5:
        metrics['cb0_entropy_first_half'] = entropy(cb0[:half])
        metrics['cb0_entropy_second_half'] = entropy(cb0[half:])
        metrics['entropy_drop'] = (metrics['cb0_entropy_first_half'] -
                                    metrics['cb0_entropy_second_half'])
    else:
        metrics['cb0_entropy_first_half'] = entropy(cb0)
        metrics['cb0_entropy_second_half'] = entropy(cb0)
        metrics['entropy_drop'] = 0.0

    # Sub-code entropy (average across groups)
    sub_entropies = []
    for g in range(n_groups - 1):
        sub_entropies.append(entropy(sub[:, g]))
    metrics['sub_entropy_mean'] = float(np.mean(sub_entropies))

    return metrics


def analyze_audio(samples, sample_rate):
    """Analyze audio waveform for quality issues. Returns dict of metrics."""
    duration = len(samples) / sample_rate
    metrics = {
        'duration_s': duration,
        'n_samples': len(samples),
        'peak': float(np.max(np.abs(samples))),
        'rms': float(np.sqrt(np.mean(samples ** 2))),
    }

    # RMS in 0.5s windows
    window_samples = sample_rate // 2
    n_windows = len(samples) // window_samples
    if n_windows > 0:
        rms_windows = []
        for i in range(n_windows):
            w = samples[i * window_samples:(i + 1) * window_samples]
            rms_windows.append(float(np.sqrt(np.mean(w ** 2))))
        metrics['rms_min_window'] = min(rms_windows)
        metrics['rms_max_window'] = max(rms_windows)
        metrics['rms_std_window'] = float(np.std(rms_windows))

        # Detect silence (RMS < 0.005 for any 0.5s window)
        silence_windows = sum(1 for r in rms_windows if r < 0.005)
        metrics['silence_windows'] = silence_windows
        metrics['n_windows'] = n_windows

        # Detect clipping (peak > 0.99)
        metrics['clipping'] = float(np.mean(np.abs(samples) > 0.99))

        # RMS ratio (last quarter vs first quarter)
        q1 = rms_windows[:len(rms_windows) // 4]
        q4 = rms_windows[3 * len(rms_windows) // 4:]
        if q1 and q4:
            metrics['rms_ratio_q4_q1'] = np.mean(q4) / max(np.mean(q1), 1e-10)
    else:
        metrics['rms_min_window'] = metrics['rms']
        metrics['rms_max_window'] = metrics['rms']
        metrics['silence_windows'] = 0
        metrics['n_windows'] = 0

    return metrics


def synthesize(server_url, text, seed, codes_path):
    """Send TTS request, return (wav_path, elapsed_s)."""
    payload = json.dumps({
        'input': text,
        'voice': 'alloy',
        'seed': seed,
    }).encode()

    # Clear old codes file
    if os.path.exists(codes_path):
        os.remove(codes_path)

    req = Request(f"{server_url}/v1/audio/speech",
                  data=payload,
                  headers={'Content-Type': 'application/json'})

    t0 = time.time()
    resp = urlopen(req, timeout=300)
    elapsed = time.time() - t0
    wav_data = resp.read()

    wav_path = codes_path.replace('.txt', '.wav')
    with open(wav_path, 'wb') as f:
        f.write(wav_data)

    return wav_path, elapsed


def run_test(server_url, seed, output_dir, codes_path):
    """Run all test cases and return results."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for name, text in TEST_CASES:
        print(f"\n--- {name} ({len(text)} chars) ---")
        print(f"  Text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

        wav_path = os.path.join(output_dir, f"{name}.wav")
        case_codes_path = os.path.join(output_dir, f"{name}_codes.txt")

        # Synthesize (server writes codes to the env-var path, we rename after)
        try:
            _, elapsed = synthesize(server_url, text, seed, codes_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'name': name, 'error': str(e)})
            continue

        # Move the codes dump to case-specific path
        if os.path.exists(codes_path):
            os.replace(codes_path, case_codes_path)

        # Move wav
        generic_wav = codes_path.replace('.txt', '.wav')
        if os.path.exists(generic_wav):
            os.replace(generic_wav, wav_path)

        # Analyze
        result = {'name': name, 'text_len': len(text), 'synth_time_s': elapsed}

        codes = load_codes(case_codes_path)
        if codes is not None:
            code_metrics = analyze_codes(codes)
            result.update(code_metrics)
            print(f"  Steps: {code_metrics['n_steps']}, "
                  f"CB0 unique: {code_metrics['cb0_unique']}, "
                  f"repeat rate: {code_metrics['cb0_repeat_rate']:.2%}, "
                  f"max window: {code_metrics['cb0_max_window_repeat']:.2%}")
            print(f"  Entropy: first={code_metrics['cb0_entropy_first_half']:.2f}, "
                  f"second={code_metrics['cb0_entropy_second_half']:.2f}, "
                  f"drop={code_metrics['entropy_drop']:+.2f} bits")

        try:
            samples, sr = read_wav_samples(wav_path)
            audio_metrics = analyze_audio(samples, sr)
            result.update(audio_metrics)
            print(f"  Audio: {audio_metrics['duration_s']:.1f}s, "
                  f"RMS={audio_metrics['rms']:.4f}, "
                  f"peak={audio_metrics['peak']:.4f}")
            if audio_metrics.get('silence_windows', 0) > 0:
                print(f"  WARNING: {audio_metrics['silence_windows']} silence windows "
                      f"(of {audio_metrics['n_windows']})")
            if audio_metrics.get('clipping', 0) > 0.001:
                print(f"  WARNING: {audio_metrics['clipping']:.2%} clipping")
        except Exception as e:
            print(f"  Audio analysis error: {e}")

        print(f"  Synthesis time: {elapsed:.1f}s")
        results.append(result)

    return results


def print_summary(results):
    """Print summary table of all results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Case':<12} {'Chars':>5} {'Steps':>5} {'EOS?':>5} "
          f"{'Duration':>8} {'CB0 Rep':>8} {'Ent Drop':>9} "
          f"{'RMS':>6} {'Silence':>8} {'Time':>6}")
    print("-" * 80)

    for r in results:
        if 'error' in r:
            print(f"{r['name']:<12} ERROR: {r['error']}")
            continue

        n_steps = r.get('n_steps', '?')
        max_steps_hit = n_steps == 200  # default max
        eos = 'no' if max_steps_hit else 'yes'
        dur = r.get('duration_s', 0)
        repeat = r.get('cb0_repeat_rate', 0)
        ent_drop = r.get('entropy_drop', 0)
        rms = r.get('rms', 0)
        silence = r.get('silence_windows', 0)
        syn_time = r.get('synth_time_s', 0)

        # Flag issues
        flags = []
        if repeat > 0.15:
            flags.append('HIGH-REPEAT')
        if ent_drop > 1.0:
            flags.append('ENTROPY-DROP')
        if silence > 0:
            flags.append('SILENCE')
        if r.get('clipping', 0) > 0.001:
            flags.append('CLIP')

        flag_str = ' ' + ','.join(flags) if flags else ''

        print(f"{r['name']:<12} {r.get('text_len', '?'):>5} {n_steps:>5} {eos:>5} "
              f"{dur:>7.1f}s {repeat:>7.1%} {ent_drop:>+8.2f}b "
              f"{rms:>6.4f} {silence:>8} {syn_time:>5.1f}s{flag_str}")

    # Quality assessment
    print("\nQuality checks:")
    issues = []
    for r in results:
        if 'error' in r:
            continue
        name = r['name']
        if r.get('cb0_repeat_rate', 0) > 0.15:
            issues.append(f"  {name}: CB0 repeat rate {r['cb0_repeat_rate']:.1%} (>15%)")
        if r.get('entropy_drop', 0) > 1.0:
            issues.append(f"  {name}: entropy drop {r['entropy_drop']:+.2f} bits (>1.0)")
        if r.get('cb0_max_window_repeat', 0) > 0.5:
            issues.append(f"  {name}: max window repeat {r['cb0_max_window_repeat']:.1%} (>50%)")
        if r.get('silence_windows', 0) > 0:
            issues.append(f"  {name}: {r['silence_windows']} silence windows")
        rr = r.get('rms_ratio_q4_q1', 1.0)
        if rr < 0.3 or rr > 3.0:
            issues.append(f"  {name}: RMS ratio Q4/Q1 = {rr:.2f} (volume inconsistency)")

    if issues:
        print("  ISSUES FOUND:")
        for iss in issues:
            print(iss)
    else:
        print("  All checks passed - no degeneration detected.")


def main():
    parser = argparse.ArgumentParser(description='TTS Long Audio Quality Test')
    parser.add_argument('--server-url', default='http://localhost:8090',
                        help='Server URL (default: http://localhost:8090)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for determinism (default: 42)')
    parser.add_argument('--output-dir', default='tts_long_test',
                        help='Output directory for WAV files and codec dumps')
    parser.add_argument('--start-server', metavar='MODEL_DIR',
                        help='Auto-start server with given model dir')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 when auto-starting server')
    parser.add_argument('--model-size', default='auto',
                        help='Model label for output (auto, 0.6b, 1.7b)')
    args = parser.parse_args()

    output_dir = args.output_dir
    codes_path = os.path.join(output_dir, '_codes_tmp.txt')
    os.makedirs(output_dir, exist_ok=True)

    server_proc = None
    if args.start_server:
        print(f"Starting server with model: {args.start_server}")
        env = os.environ.copy()
        env['TTS_DUMP_CODES'] = os.path.abspath(codes_path)
        # Use absolute path for the executable
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exe = os.path.join(script_dir, 'bin', 'local-ai-server.exe')
        if not os.path.exists(exe):
            exe = os.path.join(script_dir, 'bin', 'local-ai-server')
        cmd = [
            exe,
            f'--tts-model={args.start_server}',
            '--verbose',
            '--port=8090',
        ]
        if args.fp16:
            cmd.append('--fp16')
        server_proc = subprocess.Popen(cmd, env=env)
        print("Waiting for server to start...")
        for _ in range(60):
            time.sleep(1)
            try:
                urlopen(f"{args.server_url}/health", timeout=2)
                print("Server ready.")
                break
            except Exception:
                pass
        else:
            print("ERROR: Server failed to start within 60s")
            server_proc.kill()
            sys.exit(1)
    else:
        # Verify server is running
        try:
            urlopen(f"{args.server_url}/health", timeout=5)
        except Exception:
            print(f"ERROR: Server not reachable at {args.server_url}")
            print("Start with: TTS_DUMP_CODES=tts_long_test/_codes_tmp.txt "
                  "bin/local-ai-server.exe --tts-model=<dir> --fp16 --verbose")
            sys.exit(1)

    try:
        print(f"\nTTS Long Audio Quality Test (seed={args.seed})")
        print(f"Server: {args.server_url}")
        print(f"Output: {output_dir}/")

        results = run_test(args.server_url, args.seed, output_dir, codes_path)
        print_summary(results)

        # Save results JSON
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results: {results_path}")

    finally:
        if server_proc:
            print("Stopping server...")
            server_proc.terminate()
            server_proc.wait(timeout=10)


if __name__ == '__main__':
    main()
