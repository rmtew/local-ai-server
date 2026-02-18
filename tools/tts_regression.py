#!/usr/bin/env python3
"""
TTS regression harness for local-ai-server.

Usage examples:
  # Generate missing reference WAVs (first-time setup)
  python tts_regression.py --generate-missing

  # Refresh all reference WAVs (after intentional changes)
  python tts_regression.py --refresh-refs

  # Run regression checks against stored references
  python tts_regression.py

  # Sanity-only checks (no reference comparison)
  python tts_regression.py --sanity-only

  # Stream regression: test SSE streaming, compare WAV vs references
  python tts_regression.py --stream

Requires a running local-ai-server with --tts-model loaded.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

# ---- ANSI colors (auto-disabled when stdout is not a tty) ----

_USE_COLOR = (
    hasattr(sys.stdout, "isatty")
    and sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
)


def _sgr(code: str) -> str:
    return f"\033[{code}m" if _USE_COLOR else ""


C_RESET = _sgr("0")
C_BOLD = _sgr("1")
C_DIM = _sgr("2")
C_RED = _sgr("31")
C_GREEN = _sgr("32")
C_YELLOW = _sgr("33")
C_CYAN = _sgr("36")
C_BRED = _sgr("1;31")
C_BGREEN = _sgr("1;32")
C_BYELLOW = _sgr("1;33")
C_BCYAN = _sgr("1;36")
C_BWHITE = _sgr("1;37")


# ---- Test case definitions ----

class TestCase(NamedTuple):
    id: str
    text: str
    min_duration_s: float
    max_duration_s: float
    purpose: str


TEST_CASES: List[TestCase] = [
    TestCase(
        id="short_hello",
        text="Hello, world.",
        min_duration_s=0.3,
        max_duration_s=5.0,
        purpose="Basic short phrase",
    ),
    TestCase(
        id="medium_fox",
        text="The quick brown fox jumps over the lazy dog near the riverbank.",
        min_duration_s=1.0,
        max_duration_s=15.0,
        purpose="Standard medium sentence",
    ),
    TestCase(
        id="long_mixed",
        text='On January 15th, 2024, Dr. Smith said: "The temperature was -3.5 degrees!" Can you believe it?',
        min_duration_s=2.0,
        max_duration_s=20.0,
        purpose="Numbers, punctuation, quotes",
    ),
]

SEED = 42


# ---- WAV helpers ----

def read_wav_samples(data: bytes) -> Optional[List[float]]:
    """Parse a 16-bit PCM WAV file and return samples as floats in [-1, 1]."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None

    # Find 'fmt ' chunk
    pos = 12
    fmt_found = False
    channels = 1
    sample_rate = 24000
    bits_per_sample = 16

    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                return None
            audio_format = struct.unpack_from("<H", data, pos + 8)[0]
            if audio_format != 1:  # PCM only
                return None
            channels = struct.unpack_from("<H", data, pos + 10)[0]
            sample_rate = struct.unpack_from("<I", data, pos + 12)[0]
            bits_per_sample = struct.unpack_from("<H", data, pos + 22)[0]
            fmt_found = True
        elif chunk_id == b"data" and fmt_found:
            sample_data = data[pos + 8 : pos + 8 + chunk_size]
            if bits_per_sample != 16:
                return None
            n_samples = len(sample_data) // (2 * channels)
            samples = []
            for i in range(n_samples):
                val = struct.unpack_from("<h", sample_data, i * 2 * channels)[0]
                samples.append(val / 32768.0)
            return samples
        pos += 8 + chunk_size
        if pos % 2 == 1:  # chunks are word-aligned
            pos += 1

    return None


def wav_duration_s(data: bytes, sample_rate: int = 24000) -> float:
    """Get WAV duration in seconds."""
    samples = read_wav_samples(data)
    if samples is None:
        return 0.0
    return len(samples) / sample_rate


# ---- Comparison metrics ----

def pearson_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation coefficient between two equal-length sequences."""
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0

    mean_a = sum(a) / n
    mean_b = sum(b) / n

    cov = 0.0
    var_a = 0.0
    var_b = 0.0
    for i in range(n):
        da = a[i] - mean_a
        db = b[i] - mean_b
        cov += da * db
        var_a += da * da
        var_b += db * db

    if var_a == 0.0 or var_b == 0.0:
        return 1.0 if var_a == var_b == 0.0 else 0.0

    return cov / math.sqrt(var_a * var_b)


def snr_db(reference: Sequence[float], test: Sequence[float]) -> float:
    """Signal-to-noise ratio in dB, treating the difference as noise."""
    n = len(reference)
    if n == 0 or n != len(test):
        return 0.0

    signal_power = sum(x * x for x in reference) / n
    noise_power = sum((reference[i] - test[i]) ** 2 for i in range(n)) / n

    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return 0.0

    return 10.0 * math.log10(signal_power / noise_power)


def rms(samples: Sequence[float]) -> float:
    """Root mean square of a sample sequence."""
    if not samples:
        return 0.0
    return math.sqrt(sum(x * x for x in samples) / len(samples))


# ---- Server communication ----

def tts_request(
    server: str, text: str, seed: int, timeout_s: int,
    temperature: float = 0.3,
) -> Optional[bytes]:
    """Send a TTS request and return WAV bytes, or None on error."""
    payload = json.dumps({
        "input": text,
        "voice": "alloy",
        "seed": seed,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{server}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  {C_RED}HTTP {e.code}: {body[:200]}{C_RESET}")
        return None
    except urllib.error.URLError as e:
        print(f"  {C_RED}Cannot connect to {server}: {e.reason}{C_RESET}")
        return None
    except Exception as e:
        print(f"  {C_RED}Request error: {e}{C_RESET}")
        return None


class StreamResult(NamedTuple):
    """Result of a streaming TTS request."""
    wav_data: bytes                 # Decoded WAV from base64
    n_decode_events: int            # Number of "decoding" progress events
    final_step: int                 # Last step value from decoding events
    max_steps: int                  # max_steps from decoding events
    had_vocoder_event: bool         # Whether "vocoder" phase event was seen
    had_done_sentinel: bool         # Whether [DONE] sentinel was seen
    event_order_ok: bool            # Whether events were in correct order


def tts_request_stream(
    server: str, text: str, seed: int, timeout_s: int,
    temperature: float = 0.3,
) -> Optional[StreamResult]:
    """Send a streaming TTS request, parse SSE events, return decoded WAV
    and protocol metadata. Returns None on connection/protocol error."""
    if server.startswith("http://"):
        host_port = server[7:]
    else:
        host_port = server
    if ":" in host_port:
        host, port_str = host_port.split(":", 1)
        port = int(port_str)
    else:
        host, port = host_port, 80

    payload = json.dumps({
        "input": text,
        "voice": "alloy",
        "seed": seed,
        "stream": True,
        "temperature": temperature,
    }).encode("utf-8")

    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        conn.request("POST", "/v1/audio/speech", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
    except Exception as e:
        print(f"  {C_RED}Stream connect error: {e}{C_RESET}")
        return None

    if resp.status != 200:
        body = resp.read().decode("utf-8", errors="replace")
        print(f"  {C_RED}Stream HTTP {resp.status}: {body[:200]}{C_RESET}")
        conn.close()
        return None

    ct = resp.getheader("Content-Type", "")
    if "text/event-stream" not in ct:
        print(f"  {C_RED}Not SSE: Content-Type={ct}{C_RESET}")
        resp.read()
        conn.close()
        return None

    # Parse SSE events
    decode_steps = []
    max_steps_val = 0
    had_vocoder = False
    done_event = None
    had_done_sentinel = False
    phases_seen = []  # for order checking

    buf = b""
    while True:
        chunk = resp.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            event_data, buf = buf.split(b"\n\n", 1)
            for line in event_data.split(b"\n"):
                line_str = line.decode("utf-8", errors="replace")
                if not line_str.startswith("data: "):
                    continue
                payload_str = line_str[6:]
                if payload_str == "[DONE]":
                    had_done_sentinel = True
                    phases_seen.append("[DONE]")
                    continue
                try:
                    obj = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                phase = obj.get("phase", "")
                phases_seen.append(phase)
                if phase == "decoding":
                    decode_steps.append(obj.get("step", 0))
                    max_steps_val = obj.get("max_steps", max_steps_val)
                elif phase == "vocoder":
                    had_vocoder = True
                elif phase == "done":
                    done_event = obj

    conn.close()

    if done_event is None or "audio" not in done_event:
        print(f"  {C_RED}Stream: no done event with audio{C_RESET}")
        return None

    try:
        wav_data = base64.b64decode(done_event["audio"])
    except Exception as e:
        print(f"  {C_RED}Base64 decode failed: {e}{C_RESET}")
        return None

    # Check event ordering: all decoding before vocoder before done
    order_ok = True
    seen_vocoder = False
    seen_done = False
    for p in phases_seen:
        if p == "decoding" and (seen_vocoder or seen_done):
            order_ok = False
        elif p == "vocoder":
            seen_vocoder = True
            if seen_done:
                order_ok = False
        elif p == "done":
            seen_done = True

    return StreamResult(
        wav_data=wav_data,
        n_decode_events=len(decode_steps),
        final_step=decode_steps[-1] if decode_steps else 0,
        max_steps=max_steps_val,
        had_vocoder_event=had_vocoder,
        had_done_sentinel=had_done_sentinel,
        event_order_ok=order_ok,
    )


def check_server(server: str, timeout_s: int = 5) -> bool:
    """Check if the server is reachable."""
    try:
        req = urllib.request.Request(f"{server}/health")
        with urllib.request.urlopen(req, timeout=timeout_s):
            return True
    except Exception:
        return False


def check_tts_available(server: str, timeout_s: int = 60) -> bool:
    """Check if TTS is loaded on the server (not 501)."""
    payload = json.dumps({"input": "test", "seed": 0}).encode("utf-8")
    req = urllib.request.Request(
        f"{server}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s):
            return True
    except urllib.error.HTTPError as e:
        if e.code == 501:
            return False
        return True  # other errors mean TTS is loaded but something else failed
    except Exception:
        return False


# ---- Formatting ----

def fmt_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}m{s:.0f}s"


# ---- Core test logic ----

def run_sanity(
    tc: TestCase,
    wav_data: bytes,
    idx: int,
    total: int,
    verbose: bool,
) -> bool:
    """Run sanity checks on a WAV response (non-silence, duration bounds)."""
    samples = read_wav_samples(wav_data)
    if samples is None:
        print(f"  {C_RED}Invalid WAV data{C_RESET}")
        return False

    duration = len(samples) / 24000.0
    sample_rms = rms(samples)

    ok = True
    issues = []

    if sample_rms <= 0.001:
        issues.append(f"silent (RMS={sample_rms:.6f})")
        ok = False

    if duration < tc.min_duration_s:
        issues.append(f"too short ({duration:.2f}s < {tc.min_duration_s:.1f}s)")
        ok = False

    if duration > tc.max_duration_s:
        issues.append(f"too long ({duration:.2f}s > {tc.max_duration_s:.1f}s)")
        ok = False

    if verbose:
        print(
            f"  sanity: duration={duration:.2f}s, "
            f"RMS={sample_rms:.6f}, samples={len(samples)}"
        )

    if issues:
        print(f"  {C_RED}sanity issues: {', '.join(issues)}{C_RESET}")

    return ok


def run_comparison(
    tc: TestCase,
    wav_data: bytes,
    ref_path: Path,
    min_correlation: float,
    min_snr_db: float,
    verbose: bool,
) -> tuple[bool, str]:
    """Compare WAV against reference. Returns (passed, metrics_str)."""
    if not ref_path.exists():
        return False, "reference missing"

    ref_data = ref_path.read_bytes()
    ref_samples = read_wav_samples(ref_data)
    test_samples = read_wav_samples(wav_data)

    if ref_samples is None:
        return False, f"{C_RED}corrupt reference WAV{C_RESET}"
    if test_samples is None:
        return False, f"{C_RED}invalid test WAV{C_RESET}"

    # Sample count check
    count_match = len(test_samples) == len(ref_samples)

    # For correlation/SNR, need same length
    if len(test_samples) != len(ref_samples):
        min_len = min(len(test_samples), len(ref_samples))
        corr = pearson_correlation(ref_samples[:min_len], test_samples[:min_len])
        snr_val = snr_db(ref_samples[:min_len], test_samples[:min_len])
    else:
        corr = pearson_correlation(ref_samples, test_samples)
        snr_val = snr_db(ref_samples, test_samples)

    # Format metrics string
    count_color = C_GREEN if count_match else C_RED
    corr_color = C_GREEN if corr >= min_correlation else C_RED
    snr_color = C_GREEN if snr_val >= min_snr_db else C_RED

    snr_str = f"{snr_val:.1f}" if not math.isinf(snr_val) else "inf"

    metrics = (
        f"samples {count_color}{len(test_samples)}/{len(ref_samples)}{C_RESET}"
        f" | corr {corr_color}{corr:.6f}{C_RESET}"
        f" | SNR {snr_color}{snr_str}{C_RESET} dB"
    )

    if verbose:
        test_rms_val = rms(test_samples)
        ref_rms_val = rms(ref_samples)
        print(f"  test RMS={test_rms_val:.6f}, ref RMS={ref_rms_val:.6f}")

    passed = count_match and corr >= min_correlation and snr_val >= min_snr_db
    return passed, metrics


def generate_reference(
    tc: TestCase,
    server: str,
    samples_dir: Path,
    timeout_s: int,
    idx: int,
    total: int,
) -> bool:
    """Generate a reference WAV for a test case."""
    ref_path = samples_dir / f"{tc.id}.wav"
    tag = f"{C_BCYAN}[GEN  {idx + 1}/{total}]{C_RESET}"
    print(f"{tag} {C_BWHITE}{tc.id}{C_RESET} ...", end="", flush=True)

    t0 = time.monotonic()
    wav_data = tts_request(server, tc.text, SEED, timeout_s)
    elapsed = time.monotonic() - t0

    if wav_data is None:
        print(f"\r{tag} {C_BWHITE}{tc.id}{C_RESET} | {C_RED}FAILED{C_RESET}")
        return False

    samples = read_wav_samples(wav_data)
    if samples is None:
        print(
            f"\r{tag} {C_BWHITE}{tc.id}{C_RESET}"
            f" | {C_RED}invalid WAV response{C_RESET}"
        )
        return False

    ref_path.write_bytes(wav_data)
    duration = len(samples) / 24000.0
    print(
        f"\r{tag} {C_BWHITE}{tc.id}{C_RESET}"
        f" | {len(samples)} samples, {duration:.2f}s"
        f" | {C_DIM}{fmt_time(elapsed)}{C_RESET}"
    )
    return True


# ---- Main test runner ----

def run_regression(
    cases: List[TestCase],
    server: str,
    samples_dir: Path,
    timeout_s: int,
    min_correlation: float,
    min_snr: float,
    generate_missing: bool,
    sanity_only: bool,
    skip_sanity: bool,
    verbose: bool,
) -> int:
    """Run regression tests. Returns number of failures."""
    total = len(cases)
    failures = 0

    for idx, tc in enumerate(cases):
        ref_path = samples_dir / f"{tc.id}.wav"
        tag_start = f"{C_BCYAN}[START {idx + 1}/{total}]{C_RESET}"

        # Check if reference exists for non-sanity mode
        if not sanity_only and not ref_path.exists():
            if generate_missing:
                ok = generate_reference(tc, server, samples_dir, timeout_s, idx, total)
                if not ok:
                    failures += 1
                continue
            else:
                tag_skip = f"{C_BYELLOW}[SKIP  {idx + 1}/{total}]{C_RESET}"
                print(
                    f"{tag_skip} {C_BWHITE}{tc.id}{C_RESET}"
                    f" | {C_YELLOW}no reference (use --generate-missing){C_RESET}"
                )
                continue

        print(f"{tag_start} {C_BWHITE}{tc.id}{C_RESET} ...", end="", flush=True)

        t0 = time.monotonic()
        wav_data = tts_request(server, tc.text, SEED, timeout_s)
        elapsed = time.monotonic() - t0

        if wav_data is None:
            tag_fail = f"{C_BRED}[DONE: FAIL {idx + 1}/{total}]{C_RESET}"
            print(
                f"\r{tag_fail} {C_BWHITE}{tc.id}{C_RESET}"
                f" | {C_RED}request failed{C_RESET}"
                f" | {C_DIM}{fmt_time(elapsed)}{C_RESET}"
            )
            failures += 1
            continue

        # Sanity checks
        sanity_ok = True
        if not skip_sanity:
            sanity_ok = run_sanity(tc, wav_data, idx, total, verbose)

        if sanity_only:
            if sanity_ok:
                tag_ok = f"{C_BGREEN}[DONE: OK   {idx + 1}/{total}]{C_RESET}"
                samples = read_wav_samples(wav_data)
                n = len(samples) if samples else 0
                print(
                    f"\r{tag_ok} {C_BWHITE}{tc.id}{C_RESET}"
                    f" | {n} samples, {n / 24000.0:.2f}s"
                    f" | {C_DIM}{fmt_time(elapsed)}{C_RESET}"
                )
            else:
                tag_fail = f"{C_BRED}[DONE: FAIL {idx + 1}/{total}]{C_RESET}"
                print(
                    f"\r{tag_fail} {C_BWHITE}{tc.id}{C_RESET}"
                    f" | {C_RED}sanity check failed{C_RESET}"
                    f" | {C_DIM}{fmt_time(elapsed)}{C_RESET}"
                )
                failures += 1
            continue

        # Regression comparison
        passed, metrics = run_comparison(
            tc, wav_data, ref_path, min_correlation, min_snr, verbose
        )

        if passed and sanity_ok:
            tag_ok = f"{C_BGREEN}[DONE: OK   {idx + 1}/{total}]{C_RESET}"
            print(
                f"\r{tag_ok} {C_BWHITE}{tc.id}{C_RESET}"
                f" | {metrics}"
                f" | {C_DIM}{fmt_time(elapsed)}{C_RESET}"
            )
        else:
            tag_fail = f"{C_BRED}[DONE: FAIL {idx + 1}/{total}]{C_RESET}"
            reason = metrics
            if not sanity_ok:
                reason += f" | {C_RED}sanity failed{C_RESET}"
            print(
                f"\r{tag_fail} {C_BWHITE}{tc.id}{C_RESET}"
                f" | {reason}"
                f" | {C_DIM}{fmt_time(elapsed)}{C_RESET}"
            )
            failures += 1

    return failures


# ---- CLI ----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TTS regression test suite for local-ai-server"
    )

    p.add_argument(
        "--server",
        default="http://localhost:8090",
        help="Server URL (default: http://localhost:8090)",
    )
    p.add_argument(
        "--samples-dir",
        default="tts_samples",
        help="Reference WAV directory (default: tts_samples)",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=120,
        help="Per-request timeout in seconds (default: 120)",
    )
    p.add_argument(
        "--min-correlation",
        type=float,
        default=0.999,
        help="Minimum Pearson correlation (default: 0.999)",
    )
    p.add_argument(
        "--min-snr-db",
        type=float,
        default=60.0,
        help="Minimum SNR in dB (default: 60.0)",
    )

    # Modes
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate-missing",
        action="store_true",
        help="Generate missing reference WAVs",
    )
    mode.add_argument(
        "--refresh-refs",
        action="store_true",
        help="Regenerate all reference WAVs",
    )
    mode.add_argument(
        "--sanity-only",
        action="store_true",
        help="Sanity checks only (no reference comparison)",
    )

    p.add_argument(
        "--stream",
        action="store_true",
        help="Test streaming (SSE) mode: verify protocol + byte-identical WAV vs references",
    )
    p.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip sanity checks during regression",
    )
    p.add_argument(
        "--case",
        action="append",
        dest="cases",
        metavar="ID",
        help="Run only specific case (repeatable)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed metrics",
    )

    return p.parse_args()


def run_stream_regression(
    cases: List[TestCase],
    server: str,
    samples_dir: Path,
    timeout_s: int,
    verbose: bool,
) -> int:
    """Run streaming regression: for each case, request with stream=true and
    compare the decoded WAV against the stored reference. Also validates SSE
    protocol correctness. Returns number of failures."""
    total = len(cases)
    failures = 0

    for idx, tc in enumerate(cases):
        ref_path = samples_dir / f"{tc.id}.wav"
        tag_start = f"{C_BCYAN}[STREAM {idx + 1}/{total}]{C_RESET}"

        if not ref_path.exists():
            tag_skip = f"{C_BYELLOW}[SKIP   {idx + 1}/{total}]{C_RESET}"
            print(
                f"{tag_skip} {C_BWHITE}{tc.id}{C_RESET}"
                f" | {C_YELLOW}no reference (run without --stream first){C_RESET}"
            )
            continue

        print(f"{tag_start} {C_BWHITE}{tc.id}{C_RESET} ...", end="", flush=True)

        t0 = time.monotonic()
        sr = tts_request_stream(server, tc.text, SEED, timeout_s)
        elapsed = time.monotonic() - t0

        if sr is None:
            tag_fail = f"{C_BRED}[FAIL   {idx + 1}/{total}]{C_RESET}"
            print(
                f"\r{tag_fail} {C_BWHITE}{tc.id}{C_RESET}"
                f" | {C_RED}streaming request failed{C_RESET}"
                f" | {C_DIM}{fmt_time(elapsed)}{C_RESET}"
            )
            failures += 1
            continue

        # Protocol checks
        issues = []
        if sr.n_decode_events == 0:
            issues.append("no decode events")
        if not sr.had_vocoder_event:
            issues.append("no vocoder event")
        if not sr.had_done_sentinel:
            issues.append("no [DONE]")
        if not sr.event_order_ok:
            issues.append("wrong event order")

        # Compare WAV against reference
        ref_data = ref_path.read_bytes()
        wav_match = sr.wav_data == ref_data

        if verbose:
            print()
            print(
                f"    decode events: {sr.n_decode_events}, "
                f"steps: {sr.final_step}/{sr.max_steps}, "
                f"vocoder: {sr.had_vocoder_event}, "
                f"[DONE]: {sr.had_done_sentinel}"
            )
            print(
                f"    WAV: {len(sr.wav_data)} bytes "
                f"(ref: {len(ref_data)} bytes, match: {wav_match})"
            )

        if wav_match and not issues:
            tag_ok = f"{C_BGREEN}[OK     {idx + 1}/{total}]{C_RESET}"
            print(
                f"\r{tag_ok} {C_BWHITE}{tc.id}{C_RESET}"
                f" | {C_GREEN}WAV identical{C_RESET}"
                f" | {sr.n_decode_events} events, {sr.final_step} steps"
                f" | {C_DIM}{fmt_time(elapsed)}{C_RESET}"
            )
        else:
            tag_fail = f"{C_BRED}[FAIL   {idx + 1}/{total}]{C_RESET}"
            reasons = []
            if not wav_match:
                reasons.append(
                    f"WAV differs ({len(sr.wav_data)} vs {len(ref_data)} bytes)"
                )
            reasons.extend(issues)
            print(
                f"\r{tag_fail} {C_BWHITE}{tc.id}{C_RESET}"
                f" | {C_RED}{'; '.join(reasons)}{C_RESET}"
                f" | {C_DIM}{fmt_time(elapsed)}{C_RESET}"
            )
            failures += 1

    return failures


def main() -> int:
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    samples_dir.mkdir(exist_ok=True)

    # Filter test cases
    cases = TEST_CASES
    if args.cases:
        valid_ids = {tc.id for tc in TEST_CASES}
        for cid in args.cases:
            if cid not in valid_ids:
                print(
                    f"{C_RED}Unknown test case: {cid}{C_RESET}\n"
                    f"Available: {', '.join(sorted(valid_ids))}"
                )
                return 2
        cases = [tc for tc in TEST_CASES if tc.id in args.cases]

    # Header
    mode_str = "regression"
    if args.generate_missing:
        mode_str = "generate-missing"
    elif args.refresh_refs:
        mode_str = "refresh-refs"
    elif args.sanity_only:
        mode_str = "sanity-only"
    elif args.stream:
        mode_str = "stream"

    print(f"{C_BOLD}TTS Regression Suite{C_RESET} ({mode_str})")
    print(f"Server: {args.server}")
    if not args.sanity_only:
        print(
            f"Thresholds: correlation >= {args.min_correlation}, "
            f"SNR >= {args.min_snr_db} dB"
        )
    print()

    # Check server connectivity
    if not check_server(args.server):
        print(
            f"{C_RED}Cannot connect to server at {args.server}{C_RESET}\n"
            f"Start the server with: bin/local-ai-server.exe --tts-model=<dir>"
        )
        return 2

    # Check TTS availability
    if not check_tts_available(args.server):
        print(
            f"{C_RED}TTS not available on server{C_RESET}\n"
            f"Start the server with --tts-model=<dir>"
        )
        return 2

    t_total = time.monotonic()

    # Refresh mode: delete all existing refs then generate
    if args.refresh_refs:
        for tc in cases:
            ref_path = samples_dir / f"{tc.id}.wav"
            if ref_path.exists():
                ref_path.unlink()

        failures = 0
        for idx, tc in enumerate(cases):
            ok = generate_reference(
                tc, args.server, samples_dir, args.timeout_s, idx, len(cases)
            )
            if not ok:
                failures += 1

        elapsed_total = time.monotonic() - t_total
        print()
        if failures == 0:
            print(
                f"{C_BGREEN}References regenerated: "
                f"{len(cases)}/{len(cases)} succeeded{C_RESET}"
                f"  ({fmt_time(elapsed_total)} total)"
            )
        else:
            print(
                f"{C_BRED}Reference generation: "
                f"{failures}/{len(cases)} failed{C_RESET}"
                f"  ({fmt_time(elapsed_total)} total)"
            )
        return 1 if failures else 0

    # Streaming regression mode
    if args.stream:
        failures = run_stream_regression(
            cases=cases,
            server=args.server,
            samples_dir=samples_dir,
            timeout_s=args.timeout_s,
            verbose=args.verbose,
        )

        elapsed_total = time.monotonic() - t_total
        print()
        tested = len(cases)
        passed = tested - failures
        if failures == 0:
            print(
                f"{C_BGREEN}Stream regression PASSED: "
                f"{passed}/{tested} test cases{C_RESET}"
                f"  ({fmt_time(elapsed_total)} total)"
            )
        else:
            print(
                f"{C_BRED}Stream regression FAILED: "
                f"{failures}/{tested} test cases{C_RESET}"
                f"  ({fmt_time(elapsed_total)} total)"
            )
        return 1 if failures else 0

    # Normal regression / sanity / generate-missing mode
    failures = run_regression(
        cases=cases,
        server=args.server,
        samples_dir=samples_dir,
        timeout_s=args.timeout_s,
        min_correlation=args.min_correlation,
        min_snr=args.min_snr_db,
        generate_missing=args.generate_missing,
        sanity_only=args.sanity_only,
        skip_sanity=args.skip_sanity,
        verbose=args.verbose,
    )

    elapsed_total = time.monotonic() - t_total
    print()

    if args.generate_missing:
        if failures == 0:
            print(
                f"{C_BGREEN}Reference generation complete{C_RESET}"
                f"  ({fmt_time(elapsed_total)} total)"
            )
        else:
            print(
                f"{C_BRED}Reference generation: "
                f"{failures} failed{C_RESET}"
                f"  ({fmt_time(elapsed_total)} total)"
            )
        return 1 if failures else 0

    tested = len(cases)
    passed = tested - failures

    if failures == 0:
        label = "Sanity" if args.sanity_only else "Regression"
        print(
            f"{C_BGREEN}{label} PASSED: "
            f"{passed}/{tested} test cases within threshold{C_RESET}"
            f"  ({fmt_time(elapsed_total)} total)"
        )
    else:
        label = "Sanity" if args.sanity_only else "Regression"
        print(
            f"{C_BRED}{label} FAILED: "
            f"{failures}/{tested} test cases outside threshold{C_RESET}"
            f"  ({fmt_time(elapsed_total)} total)"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
