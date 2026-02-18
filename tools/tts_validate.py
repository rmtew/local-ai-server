#!/usr/bin/env python3
"""
Comprehensive TTS validation across models, quantization modes, and voices.

Manages server lifecycle — starts the server with each config, runs all tests,
kills the server, and repeats for the next config.

Test matrix:
  Models:  0.6B CustomVoice, 0.6B Base, 1.7B Base
  Quant:   F32 (--no-fp16), FP16 (default), INT8 (--int8-tts)
  Voices:  CustomVoice: alloy, serena, ryan, aiden; Base: alloy only

Tests per (model, quant, voice):
  1. Determinism — same seed × 2 requests → byte-identical WAV
  2. Sanity — non-silent (RMS > 0.001), duration within bounds
  3. Round-trip ASR — TTS→ASR via timestamps endpoint, normalize + compare
  4. Streaming — SSE request, validate protocol events + audio non-silence

Usage:
  python tools/tts_validate.py                     # Full matrix
  python tools/tts_validate.py --model 0.6b-cv     # Single model
  python tools/tts_validate.py --quant int8         # All models, INT8 only
  python tools/tts_validate.py --quick              # Skip streaming + multi-voice
  python tools/tts_validate.py --no-1.7b            # Skip 1.7B (VRAM-constrained)
  python tools/tts_validate.py --server-only        # Test running server, no lifecycle
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import math
import os
import re
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

# ---- ANSI colors ----

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

# ---- Constants ----

SEED = 42
TEMPERATURE = 0.3
DEFAULT_PORT = 8099  # Avoid conflict with a running dev server on 8090
HEALTH_TIMEOUT_S = 60
REQUEST_TIMEOUT_S = 120

TEST_TEXT = "The quick brown fox jumps over the lazy dog."
TEST_TEXT_SHORT = "Hello, world."

# Duration bounds for sanity checks
MIN_DURATION_S = 0.3
MAX_DURATION_S = 15.0

# ---- Model definitions ----

DEPS_ROOT = os.environ.get("DEPS_ROOT", "C:/Data/R/git/claude-repos/deps")
TTS_MODELS_DIR = os.path.join(DEPS_ROOT, "models", "tts")
ASR_MODELS_DIR = os.path.join(DEPS_ROOT, "models", "qwen-asr")


class ModelDef(NamedTuple):
    key: str            # Short key for CLI (e.g. "0.6b-cv")
    label: str          # Display label (e.g. "0.6B-CV")
    dirname: str        # Directory name under TTS_MODELS_DIR
    voices: List[str]   # Voices to test
    is_1_7b: bool       # Whether this is the 1.7B model


MODELS: List[ModelDef] = [
    ModelDef(
        key="0.6b-cv",
        label="0.6B-CV",
        dirname="qwen3-tts-12hz-0.6b-customvoice",
        voices=["alloy", "serena", "ryan", "aiden"],
        is_1_7b=False,
    ),
    ModelDef(
        key="0.6b-base",
        label="0.6B-Base",
        dirname="qwen3-tts-12hz-0.6b-base",
        voices=["alloy"],
        is_1_7b=False,
    ),
    ModelDef(
        key="1.7b-base",
        label="1.7B-Base",
        dirname="qwen3-tts-12hz-1.7b-base",
        voices=["alloy"],
        is_1_7b=True,
    ),
]

QUANT_MODES = {
    "f32":  {"label": "F32",  "args": ["--no-fp16"]},
    "fp16": {"label": "FP16", "args": []},           # default
    "int8": {"label": "INT8", "args": ["--int8-tts"]},
}

# ---- WAV helpers (from tts_regression.py) ----


def read_wav_samples(data: bytes) -> Optional[List[float]]:
    """Parse a 16-bit PCM WAV file and return samples as floats in [-1, 1]."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None

    pos = 12
    fmt_found = False
    channels = 1

    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                return None
            audio_format = struct.unpack_from("<H", data, pos + 8)[0]
            if audio_format != 1:
                return None
            channels = struct.unpack_from("<H", data, pos + 10)[0]
            fmt_found = True
        elif chunk_id == b"data" and fmt_found:
            sample_data = data[pos + 8:pos + 8 + chunk_size]
            n_samples = len(sample_data) // (2 * channels)
            samples = []
            for i in range(n_samples):
                val = struct.unpack_from("<h", sample_data, i * 2 * channels)[0]
                samples.append(val / 32768.0)
            return samples
        pos += 8 + chunk_size
        if pos % 2 == 1:
            pos += 1

    return None


def wav_rms(data: bytes) -> float:
    """Compute RMS of WAV samples."""
    samples = read_wav_samples(data)
    if not samples:
        return 0.0
    return math.sqrt(sum(x * x for x in samples) / len(samples))


def wav_duration_s(data: bytes, sample_rate: int = 24000) -> float:
    """Get WAV duration in seconds."""
    samples = read_wav_samples(data)
    if samples is None:
        return 0.0
    return len(samples) / sample_rate


# ---- Text normalization for ASR round-trip ----


def normalize_text(text: str) -> List[str]:
    """Normalize text for comparison: lowercase, strip punctuation, split words."""
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text.split()


# ---- Server communication ----


def tts_request(
    server: str, text: str, voice: str, seed: int,
    timeout_s: int = REQUEST_TIMEOUT_S,
) -> Optional[bytes]:
    """Send a TTS request and return WAV bytes."""
    payload = json.dumps({
        "input": text,
        "voice": voice,
        "seed": seed,
        "temperature": TEMPERATURE,
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
    except Exception as e:
        print(f"    {C_RED}TTS request error: {e}{C_RESET}")
        return None


def tts_request_timestamps(
    server: str, text: str, voice: str, seed: int,
    timeout_s: int = REQUEST_TIMEOUT_S,
) -> Optional[dict]:
    """Send a TTS request with timestamps=true (TTS→ASR round-trip).
    Returns parsed JSON response with 'text', 'audio', 'words' fields."""
    payload = json.dumps({
        "input": text,
        "voice": voice,
        "seed": seed,
        "temperature": TEMPERATURE,
        "timestamps": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{server}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"    {C_RED}Timestamps request error: {e}{C_RESET}")
        return None


class StreamResult(NamedTuple):
    wav_data: bytes
    n_decode_events: int
    had_vocoder_event: bool
    had_done_sentinel: bool
    event_order_ok: bool


def tts_request_stream(
    server: str, text: str, voice: str, seed: int,
    timeout_s: int = REQUEST_TIMEOUT_S,
) -> Optional[StreamResult]:
    """Send a streaming TTS request, parse SSE events, return result."""
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
        "voice": voice,
        "seed": seed,
        "stream": True,
        "temperature": TEMPERATURE,
    }).encode("utf-8")

    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        conn.request("POST", "/v1/audio/speech", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
    except Exception as e:
        print(f"    {C_RED}Stream connect error: {e}{C_RESET}")
        return None

    if resp.status != 200:
        body = resp.read().decode("utf-8", errors="replace")
        print(f"    {C_RED}Stream HTTP {resp.status}: {body[:200]}{C_RESET}")
        conn.close()
        return None

    ct = resp.getheader("Content-Type", "")
    if "text/event-stream" not in ct:
        print(f"    {C_RED}Not SSE: Content-Type={ct}{C_RESET}")
        resp.read()
        conn.close()
        return None

    # Parse SSE events
    decode_count = 0
    had_vocoder = False
    done_event = None
    had_done_sentinel = False
    phases_seen = []

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
                    decode_count += 1
                elif phase == "vocoder":
                    had_vocoder = True
                elif phase == "done":
                    done_event = obj

    conn.close()

    if done_event is None or "audio" not in done_event:
        print(f"    {C_RED}Stream: no done event with audio{C_RESET}")
        return None

    try:
        wav_data = base64.b64decode(done_event["audio"])
    except Exception as e:
        print(f"    {C_RED}Base64 decode failed: {e}{C_RESET}")
        return None

    # Check event ordering
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
        n_decode_events=decode_count,
        had_vocoder_event=had_vocoder,
        had_done_sentinel=had_done_sentinel,
        event_order_ok=order_ok,
    )


def check_health(server: str, timeout_s: int = 5) -> bool:
    """Check if the server is reachable."""
    try:
        req = urllib.request.Request(f"{server}/health")
        with urllib.request.urlopen(req, timeout=timeout_s):
            return True
    except Exception:
        return False


# ---- Server lifecycle ----


def find_server_exe() -> str:
    """Find the server executable."""
    # Check relative to script location (tools/ -> bin/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    exe = project_root / "bin" / "local-ai-server.exe"
    if exe.exists():
        return str(exe)
    # Linux
    exe_linux = project_root / "bin" / "local-ai-server"
    if exe_linux.exists():
        return str(exe_linux)
    raise FileNotFoundError(
        f"Server executable not found at {exe} or {exe_linux}. Run build.bat first."
    )


def kill_server_on_port(port: int) -> None:
    """Kill any process listening on the given port (Windows)."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = int(parts[-1])
                    if pid > 0:
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                       capture_output=True, timeout=5)
        except Exception:
            pass
    else:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in result.stdout.strip().splitlines():
                try:
                    os.kill(int(pid_str), signal.SIGKILL)
                except (ValueError, ProcessLookupError):
                    pass
        except Exception:
            pass


def start_server(
    tts_model_path: str,
    asr_model_path: str,
    quant_args: List[str],
    port: int,
) -> Tuple[subprocess.Popen, str]:
    """Start the server, return (process, server_url)."""
    exe = find_server_exe()
    args = [
        exe,
        f"--tts-model={tts_model_path}",
        f"--model={asr_model_path}",
        f"--port={port}",
        "--verbose",
    ] + quant_args

    # Start server, capture stdout/stderr for VRAM parsing
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    server_url = f"http://localhost:{port}"
    return proc, server_url


def wait_for_health(server: str, timeout_s: int = HEALTH_TIMEOUT_S) -> bool:
    """Poll /health until the server is ready."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check_health(server, timeout_s=3):
            return True
        time.sleep(1.0)
    return False


def stop_server(proc: subprocess.Popen) -> str:
    """Stop the server and return captured stdout."""
    stdout_text = ""
    try:
        proc.terminate()
        try:
            # Read remaining output
            remaining, _ = proc.communicate(timeout=10)
            if remaining:
                stdout_text = remaining
        except subprocess.TimeoutExpired:
            proc.kill()
            remaining, _ = proc.communicate(timeout=5)
            if remaining:
                stdout_text = remaining
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    return stdout_text


def parse_vram_from_output(output: str) -> Optional[int]:
    """Parse VRAM usage (MB) from server startup output.
    Matches patterns like '1278 MB VRAM' or 'VRAM: 1278 MB'."""
    # Server prints e.g. "GPU: cuBLAS — 123 weights, 1278 MB VRAM"
    # and per-model lines like "853 MB VRAM (123 weights: ...)"
    matches = re.findall(r'(\d+)\s*MB\s*VRAM', output)
    if matches:
        # Last match is the GPU summary total
        return int(matches[-1])
    # Fallback: "VRAM: 1278 MB" or "VRAM 1278 MB"
    matches = re.findall(r'VRAM[:\s]+(\d+)\s*MB', output, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    return None


# ---- Collect server output in background ----


class OutputCollector:
    """Reads server stdout in background thread, stores lines."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.lines: List[str] = []
        self._thread = None

    def start(self):
        import threading
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        try:
            for line in self.proc.stdout:
                self.lines.append(line)
        except Exception:
            pass

    def get_output(self) -> str:
        if self._thread:
            self._thread.join(timeout=3)
        return "".join(self.lines)


# ---- Test implementations ----


class TestResult(NamedTuple):
    name: str
    passed: bool
    detail: str


def test_determinism(server: str, voice: str) -> TestResult:
    """Same seed × 2 requests → byte-identical WAV."""
    wav1 = tts_request(server, TEST_TEXT_SHORT, voice, SEED)
    if wav1 is None:
        return TestResult("determinism", False, "request 1 failed")

    wav2 = tts_request(server, TEST_TEXT_SHORT, voice, SEED)
    if wav2 is None:
        return TestResult("determinism", False, "request 2 failed")

    if wav1 == wav2:
        dur = wav_duration_s(wav1)
        return TestResult("determinism", True, f"identical ({len(wav1)} bytes, {dur:.2f}s)")
    else:
        return TestResult("determinism", False,
                          f"differ ({len(wav1)} vs {len(wav2)} bytes)")


def test_sanity(server: str, voice: str) -> TestResult:
    """Non-silent, duration within bounds."""
    wav = tts_request(server, TEST_TEXT, voice, SEED)
    if wav is None:
        return TestResult("sanity", False, "request failed")

    samples = read_wav_samples(wav)
    if samples is None:
        return TestResult("sanity", False, "invalid WAV")

    duration = len(samples) / 24000.0
    rms_val = math.sqrt(sum(x * x for x in samples) / len(samples)) if samples else 0.0

    issues = []
    if rms_val <= 0.001:
        issues.append(f"silent (RMS={rms_val:.6f})")
    if duration < MIN_DURATION_S:
        issues.append(f"too short ({duration:.2f}s)")
    if duration > MAX_DURATION_S:
        issues.append(f"too long ({duration:.2f}s)")

    if issues:
        return TestResult("sanity", False, ", ".join(issues))
    return TestResult("sanity", True, f"RMS={rms_val:.4f}, {duration:.2f}s")


def test_asr_roundtrip(server: str, voice: str) -> TestResult:
    """TTS→ASR via timestamps endpoint, compare normalized text."""
    resp = tts_request_timestamps(server, TEST_TEXT, voice, SEED)
    if resp is None:
        return TestResult("asr-rt", False, "request failed")

    recognized = resp.get("text", "")
    if not recognized:
        return TestResult("asr-rt", False, "empty ASR text")

    expected_words = normalize_text(TEST_TEXT)
    actual_words = normalize_text(recognized)

    if expected_words == actual_words:
        return TestResult("asr-rt", True, f'"{recognized[:60]}"')
    else:
        # Show which words differ
        return TestResult("asr-rt", False,
                          f'expected {expected_words}, got {actual_words}')


def test_streaming(server: str, voice: str) -> TestResult:
    """SSE request: validate protocol events + audio non-silence."""
    sr = tts_request_stream(server, TEST_TEXT_SHORT, voice, SEED)
    if sr is None:
        return TestResult("stream", False, "request failed")

    issues = []
    if sr.n_decode_events == 0:
        issues.append("no decode events")
    if not sr.had_vocoder_event:
        issues.append("no vocoder event")
    if not sr.had_done_sentinel:
        issues.append("no [DONE]")
    if not sr.event_order_ok:
        issues.append("wrong event order")

    # Check audio
    rms_val = wav_rms(sr.wav_data)
    if rms_val <= 0.001:
        issues.append(f"silent audio (RMS={rms_val:.6f})")

    if issues:
        return TestResult("stream", False, ", ".join(issues))
    return TestResult("stream", True,
                      f"{sr.n_decode_events} steps, RMS={rms_val:.4f}")


# ---- Result tracking ----


class ComboResult:
    """Tracks results for one (model, quant) combo across all voices."""

    def __init__(self, model_label: str, quant_label: str):
        self.model_label = model_label
        self.quant_label = quant_label
        self.vram_mb: Optional[int] = None
        self.voice_results: Dict[str, List[TestResult]] = {}
        self.voices_passed = 0
        self.voices_total = 0

    def add_voice(self, voice: str, results: List[TestResult]):
        self.voice_results[voice] = results
        self.voices_total += 1
        if all(r.passed for r in results):
            self.voices_passed += 1

    @property
    def all_passed(self) -> bool:
        return all(
            r.passed
            for results in self.voice_results.values()
            for r in results
        )

    def test_status(self, test_name: str) -> str:
        """Aggregate status for a test across all voices."""
        statuses = []
        for results in self.voice_results.values():
            for r in results:
                if r.name == test_name:
                    statuses.append(r.passed)
        if not statuses:
            return f"{C_DIM}skip{C_RESET}"
        if all(statuses):
            return f"{C_GREEN}PASS{C_RESET}"
        return f"{C_RED}FAIL{C_RESET}"


# ---- Formatting ----


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}m{s:.0f}s"


# ---- Main runner ----


def find_asr_model() -> str:
    """Find an ASR model for round-trip testing. Prefer 0.6B for speed."""
    for name in ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]:
        path = os.path.join(ASR_MODELS_DIR, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"No ASR model found in {ASR_MODELS_DIR}. "
        "Need ASR for round-trip test."
    )


def run_combo(
    model: ModelDef,
    quant_key: str,
    server_url: str,
    voices: List[str],
    skip_streaming: bool,
    skip_asr_rt: bool,
) -> ComboResult:
    """Run all tests for one (model, quant) combo. Server must be running."""
    quant = QUANT_MODES[quant_key]
    result = ComboResult(model.label, quant["label"])

    for voice in voices:
        tag = f"  {C_BCYAN}[{voice}]{C_RESET}"
        print(f"{tag} Testing voice '{voice}'...")
        voice_results: List[TestResult] = []

        # 1. Determinism
        r = test_determinism(server_url, voice)
        status = f"{C_GREEN}PASS{C_RESET}" if r.passed else f"{C_RED}FAIL{C_RESET}"
        print(f"    determinism: {status} — {r.detail}")
        voice_results.append(r)

        # 2. Sanity
        r = test_sanity(server_url, voice)
        status = f"{C_GREEN}PASS{C_RESET}" if r.passed else f"{C_RED}FAIL{C_RESET}"
        print(f"    sanity:      {status} — {r.detail}")
        voice_results.append(r)

        # 3. ASR round-trip
        if not skip_asr_rt:
            r = test_asr_roundtrip(server_url, voice)
            status = f"{C_GREEN}PASS{C_RESET}" if r.passed else f"{C_RED}FAIL{C_RESET}"
            print(f"    asr-rt:      {status} — {r.detail}")
            voice_results.append(r)

        # 4. Streaming
        if not skip_streaming:
            r = test_streaming(server_url, voice)
            status = f"{C_GREEN}PASS{C_RESET}" if r.passed else f"{C_RED}FAIL{C_RESET}"
            print(f"    stream:      {status} — {r.detail}")
            voice_results.append(r)

        result.add_voice(voice, voice_results)

    return result


def main() -> int:
    p = argparse.ArgumentParser(
        description="Comprehensive TTS validation across models, quant modes, and voices"
    )
    p.add_argument("--model", choices=[m.key for m in MODELS],
                   help="Test only this model (e.g. 0.6b-cv, 0.6b-base, 1.7b-base)")
    p.add_argument("--quant", choices=list(QUANT_MODES.keys()),
                   help="Test only this quant mode (f32, fp16, int8)")
    p.add_argument("--quick", action="store_true",
                   help="Skip streaming + multi-voice (alloy only)")
    p.add_argument("--no-1.7b", dest="no_17b", action="store_true",
                   help="Skip 1.7B model (VRAM-constrained)")
    p.add_argument("--server-only", action="store_true",
                   help="Skip server management, test whatever is running")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"Port for managed server (default: {DEFAULT_PORT})")
    p.add_argument("--server", default=None,
                   help="Server URL for --server-only mode (default: http://localhost:<port>)")
    p.add_argument("--no-asr-rt", dest="no_asr_rt", action="store_true",
                   help="Skip ASR round-trip test")
    p.add_argument("--verbose", action="store_true",
                   help="Show server startup output")

    args = p.parse_args()

    # Build test matrix
    models_to_test = MODELS
    if args.model:
        models_to_test = [m for m in MODELS if m.key == args.model]
    if args.no_17b:
        models_to_test = [m for m in models_to_test if not m.is_1_7b]

    quant_keys = list(QUANT_MODES.keys())
    if args.quant:
        quant_keys = [args.quant]

    if not models_to_test:
        print(f"{C_RED}No models selected{C_RESET}")
        return 2

    # Discover model paths
    for m in models_to_test:
        path = os.path.join(TTS_MODELS_DIR, m.dirname)
        if not os.path.isdir(path):
            print(f"{C_RED}Model not found: {path}{C_RESET}")
            return 2

    skip_asr_rt = args.no_asr_rt
    asr_model_path = None
    if not args.server_only and not skip_asr_rt:
        try:
            asr_model_path = find_asr_model()
        except FileNotFoundError as e:
            print(f"{C_YELLOW}Warning: {e}{C_RESET}")
            print(f"{C_YELLOW}Disabling ASR round-trip test{C_RESET}")
            skip_asr_rt = True

    # Header
    n_combos = len(models_to_test) * len(quant_keys)
    print(f"{C_BOLD}TTS Comprehensive Validation{C_RESET}")
    print(f"Models: {', '.join(m.label for m in models_to_test)}")
    print(f"Quant:  {', '.join(QUANT_MODES[q]['label'] for q in quant_keys)}")
    print(f"Combos: {n_combos}")
    if args.quick:
        print(f"Mode:   {C_YELLOW}quick{C_RESET} (alloy only, no streaming)")
    if args.server_only:
        print(f"Server: {C_YELLOW}external{C_RESET} (no lifecycle management)")
    print()

    all_results: List[ComboResult] = []
    t_total = time.monotonic()

    for model in models_to_test:
        tts_model_path = os.path.join(TTS_MODELS_DIR, model.dirname)

        for quant_key in quant_keys:
            quant = QUANT_MODES[quant_key]
            combo_label = f"{model.label} / {quant['label']}"
            print(f"{C_BOLD}{'=' * 60}{C_RESET}")
            print(f"{C_BWHITE}{combo_label}{C_RESET}")
            print(f"{C_BOLD}{'=' * 60}{C_RESET}")

            voices = model.voices if not args.quick else ["alloy"]
            skip_streaming = args.quick

            if args.server_only:
                # Use running server
                server_url = args.server or f"http://localhost:{args.port}"
                if not check_health(server_url):
                    print(f"  {C_RED}Server not reachable at {server_url}{C_RESET}")
                    cr = ComboResult(model.label, quant["label"])
                    all_results.append(cr)
                    continue

                cr = run_combo(model, quant_key, server_url, voices,
                               skip_streaming, skip_asr_rt)
                all_results.append(cr)
            else:
                # Managed server lifecycle
                print(f"  Starting server...")
                kill_server_on_port(args.port)
                time.sleep(0.5)

                # Build server args
                server_quant_args = list(quant["args"])
                # For F32 mode, also disable ASR FP16 for consistency
                # (but ASR INT8 is default from config, so pass explicit flags)
                if quant_key == "f32":
                    server_quant_args.append("--no-fp16-asr")

                asr_path = asr_model_path or ""
                if not asr_path and not skip_asr_rt:
                    skip_asr_rt = True

                try:
                    proc, server_url = start_server(
                        tts_model_path, asr_path,
                        server_quant_args, args.port,
                    )
                except FileNotFoundError as e:
                    print(f"  {C_RED}{e}{C_RESET}")
                    return 2

                # Collect output in background
                collector = OutputCollector(proc)
                collector.start()

                # Wait for health
                print(f"  Waiting for server...", end="", flush=True)
                if not wait_for_health(server_url, HEALTH_TIMEOUT_S):
                    print(f" {C_RED}TIMEOUT{C_RESET}")
                    output = stop_server(proc)
                    server_output = collector.get_output()
                    if args.verbose and server_output:
                        print(f"  Server output:\n{server_output[:2000]}")
                    cr = ComboResult(model.label, quant["label"])
                    all_results.append(cr)
                    continue
                print(f" {C_GREEN}ready{C_RESET}")

                # Run tests
                t_combo = time.monotonic()
                cr = run_combo(model, quant_key, server_url, voices,
                               skip_streaming, skip_asr_rt)

                # Stop server and parse output
                output = stop_server(proc)
                server_output = collector.get_output()
                cr.vram_mb = parse_vram_from_output(server_output)

                if args.verbose and server_output:
                    # Show last few lines of server output
                    lines = server_output.strip().splitlines()
                    if len(lines) > 10:
                        print(f"  {C_DIM}(server output: {len(lines)} lines, showing last 10){C_RESET}")
                        for line in lines[-10:]:
                            print(f"  {C_DIM}{line.rstrip()}{C_RESET}")
                    else:
                        for line in lines:
                            print(f"  {C_DIM}{line.rstrip()}{C_RESET}")

                elapsed_combo = time.monotonic() - t_combo
                vram_str = f"{cr.vram_mb} MB" if cr.vram_mb else "N/A"
                print(f"  VRAM: {vram_str}, elapsed: {fmt_time(elapsed_combo)}")

                all_results.append(cr)

            print()

    # ---- Summary table ----
    elapsed_total = time.monotonic() - t_total
    print(f"{C_BOLD}{'=' * 78}{C_RESET}")
    print(f"{C_BOLD}Summary{C_RESET}  ({fmt_time(elapsed_total)} total)")
    print(f"{C_BOLD}{'=' * 78}{C_RESET}")

    # Determine which test columns to show
    has_asr = not skip_asr_rt
    has_stream = not args.quick

    # Header
    hdr = f"{'Model':<15} {'Quant':<6} {'VRAM':>6}  {'Determ':<6} {'Sanity':<6}"
    if has_asr:
        hdr += f" {'ASR-RT':<6}"
    if has_stream:
        hdr += f" {'Stream':<6}"
    hdr += f"  {'Voices'}"
    print(f"{C_BOLD}{hdr}{C_RESET}")
    print("-" * 78)

    any_fail = False
    for cr in all_results:
        if not cr.all_passed:
            any_fail = True

        vram_str = f"{cr.vram_mb:>4} MB" if cr.vram_mb else "   N/A"
        determ = cr.test_status("determinism")
        sanity = cr.test_status("sanity")

        row = f"{cr.model_label:<15} {cr.quant_label:<6} {vram_str}  {determ:<15} {sanity:<15}"
        if has_asr:
            asr_rt = cr.test_status("asr-rt")
            row += f" {asr_rt:<15}"
        if has_stream:
            stream = cr.test_status("stream")
            row += f" {stream:<15}"
        row += f"  {cr.voices_passed}/{cr.voices_total}"
        print(row)

    print()
    if any_fail:
        print(f"{C_BRED}VALIDATION FAILED{C_RESET}")
        # Show details of failures
        for cr in all_results:
            for voice, results in cr.voice_results.items():
                for r in results:
                    if not r.passed:
                        print(f"  {C_RED}FAIL{C_RESET} {cr.model_label}/{cr.quant_label}"
                              f" [{voice}] {r.name}: {r.detail}")
        return 1
    else:
        print(f"{C_BGREEN}ALL VALIDATION PASSED{C_RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
