#!/usr/bin/env python3
"""
Standalone test for TTS SSE streaming.

Validates:
  1. SSE protocol: correct event framing, parseable JSON
  2. Progress events: "decoding" steps increment, max_steps is consistent
  3. Lifecycle: "vocoder" event fires after decode, "done" event has audio
  4. Audio correctness: base64 WAV decodes to valid WAV, byte-identical to
     non-streaming output with the same seed

Usage:
  python test_tts_streaming.py                    # run with defaults
  python test_tts_streaming.py --server http://localhost:8090
  python test_tts_streaming.py --verbose          # show each SSE event
  python test_tts_streaming.py --save-wav out.wav # save decoded audio

Requires a running local-ai-server with --tts-model loaded.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request


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
C_BRED = _sgr("1;31")
C_BGREEN = _sgr("1;32")
C_BYELLOW = _sgr("1;33")
C_BWHITE = _sgr("1;37")


def ok(msg: str) -> None:
    print(f"  {C_GREEN}PASS{C_RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {C_RED}FAIL{C_RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {C_DIM}{msg}{C_RESET}")


# ---- SSE parser ----

def parse_sse_events(response) -> list[dict | str]:
    """Read SSE events from an HTTP response. Returns list of parsed JSON
    objects, or raw strings for non-JSON events like '[DONE]'."""
    events = []
    buf = b""

    while True:
        chunk = response.read(4096)
        if not chunk:
            break
        buf += chunk

        # Process complete events (delimited by \n\n)
        while b"\n\n" in buf:
            event_data, buf = buf.split(b"\n\n", 1)
            for line in event_data.split(b"\n"):
                line = line.decode("utf-8", errors="replace")
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        events.append("[DONE]")
                    else:
                        try:
                            events.append(json.loads(payload))
                        except json.JSONDecodeError:
                            events.append(payload)

    # Handle any remaining data
    if buf.strip():
        for line in buf.split(b"\n"):
            line = line.decode("utf-8", errors="replace").strip()
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    events.append("[DONE]")
                else:
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        events.append(payload)

    return events


# ---- WAV validation ----

def validate_wav(data: bytes) -> tuple[bool, str]:
    """Check if data is a valid WAV file. Returns (valid, description)."""
    if len(data) < 44:
        return False, f"too short ({len(data)} bytes)"
    if data[:4] != b"RIFF":
        return False, f"bad magic: {data[:4]!r}"
    if data[8:12] != b"WAVE":
        return False, f"not WAVE format"

    fmt_code = struct.unpack_from("<H", data, 20)[0]
    if fmt_code != 1:
        return False, f"not PCM (format={fmt_code})"

    channels = struct.unpack_from("<H", data, 22)[0]
    sample_rate = struct.unpack_from("<I", data, 24)[0]
    bits = struct.unpack_from("<H", data, 34)[0]

    # Find data chunk size
    data_size = struct.unpack_from("<I", data, 40)[0]
    n_samples = data_size // (bits // 8) // channels
    duration = n_samples / sample_rate

    desc = (f"{n_samples} samples, {duration:.2f}s, "
            f"{sample_rate}Hz, {bits}-bit, {channels}ch")
    return True, desc


# ---- Main test ----

def run_test(
    server: str,
    text: str,
    seed: int,
    verbose: bool,
    save_wav: str | None,
) -> bool:
    """Run the streaming test. Returns True if all checks pass."""
    passed = True

    # Parse server URL
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
        "temperature": 0.3,
    }).encode("utf-8")

    # ---- Phase 1: Streaming request ----
    print(f"\n{C_BOLD}Phase 1: SSE streaming request{C_RESET}")
    info(f"POST {server}/v1/audio/speech (stream=true, seed={seed})")

    t0 = time.monotonic()
    conn = http.client.HTTPConnection(host, port, timeout=120)
    conn.request("POST", "/v1/audio/speech", body=payload,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()

    # Check response headers
    if resp.status != 200:
        fail(f"HTTP status {resp.status} (expected 200)")
        body = resp.read().decode("utf-8", errors="replace")
        info(f"Body: {body[:200]}")
        conn.close()
        return False
    ok(f"HTTP 200")

    ct = resp.getheader("Content-Type", "")
    if "text/event-stream" in ct:
        ok(f"Content-Type: {ct}")
    else:
        fail(f"Content-Type: {ct} (expected text/event-stream)")
        passed = False

    # Parse SSE events
    events = parse_sse_events(resp)
    elapsed = time.monotonic() - t0
    conn.close()

    info(f"Received {len(events)} SSE events in {elapsed:.1f}s")

    if not events:
        fail("No SSE events received")
        return False

    # ---- Phase 2: Validate event structure ----
    print(f"\n{C_BOLD}Phase 2: Event structure validation{C_RESET}")

    decode_events = []
    vocoder_event = None
    done_event = None
    done_sentinel = False

    for i, ev in enumerate(events):
        if verbose:
            if isinstance(ev, dict):
                info(f"  event {i}: {json.dumps(ev)[:120]}")
            else:
                info(f"  event {i}: {ev}")

        if isinstance(ev, str):
            if ev == "[DONE]":
                done_sentinel = True
            continue

        if not isinstance(ev, dict):
            fail(f"Event {i}: not a JSON object or string")
            passed = False
            continue

        phase = ev.get("phase")
        if phase == "decoding":
            decode_events.append(ev)
        elif phase == "vocoder":
            vocoder_event = ev
        elif phase == "done":
            done_event = ev
        elif phase is None:
            # Could be an error event
            if "error" in ev:
                fail(f"Server error: {ev['error']}")
                return False
        else:
            fail(f"Unknown phase: {phase}")
            passed = False

    # Check decoding events
    if decode_events:
        ok(f"{len(decode_events)} decoding events")

        # Steps should increment
        steps = [e.get("step", -1) for e in decode_events]
        max_steps_vals = [e.get("max_steps", -1) for e in decode_events]

        if steps == sorted(steps) and steps[0] >= 1:
            ok(f"Steps increment correctly: {steps[0]}..{steps[-1]}")
        else:
            fail(f"Steps not monotonically increasing: {steps[:10]}...")
            passed = False

        # max_steps should be consistent
        if len(set(max_steps_vals)) == 1:
            ok(f"max_steps consistent: {max_steps_vals[0]}")
        else:
            fail(f"max_steps inconsistent: {set(max_steps_vals)}")
            passed = False

        # Each event should have both fields
        for j, e in enumerate(decode_events):
            if "step" not in e or "max_steps" not in e:
                fail(f"Decoding event {j} missing step/max_steps fields")
                passed = False
                break
    else:
        fail("No decoding events received")
        passed = False

    # Check vocoder event
    if vocoder_event is not None:
        ok("Vocoder event received")
    else:
        fail("No vocoder event received")
        passed = False

    # Check done event
    if done_event is not None:
        ok("Done event received")
        for field in ("n_steps", "n_samples", "elapsed_ms", "audio"):
            if field in done_event:
                if field != "audio":
                    ok(f"  {field}: {done_event[field]}")
            else:
                fail(f"  Missing field: {field}")
                passed = False
    else:
        fail("No done event received")
        return False

    # Check [DONE] sentinel
    if done_sentinel:
        ok("[DONE] sentinel received")
    else:
        fail("[DONE] sentinel missing")
        passed = False

    # Check event ordering
    event_phases = []
    for ev in events:
        if isinstance(ev, dict) and "phase" in ev:
            event_phases.append(ev["phase"])
        elif ev == "[DONE]":
            event_phases.append("[DONE]")

    # All decoding events should come before vocoder, vocoder before done
    decode_done = False
    vocoder_done = False
    order_ok = True
    for p in event_phases:
        if p == "decoding":
            if vocoder_done or (done_event and p == "done"):
                order_ok = False
        elif p == "vocoder":
            decode_done = True
            if done_event is not None and vocoder_done:
                order_ok = False
            vocoder_done = True
        elif p == "done":
            if not vocoder_done:
                order_ok = False

    if order_ok:
        ok("Event ordering: decoding* -> vocoder -> done -> [DONE]")
    else:
        fail(f"Event ordering wrong: {event_phases[:20]}")
        passed = False

    # ---- Phase 3: Decode and validate audio ----
    print(f"\n{C_BOLD}Phase 3: Audio validation{C_RESET}")

    audio_b64 = done_event.get("audio", "")
    if not audio_b64:
        fail("No audio data in done event")
        return False

    try:
        wav_data = base64.b64decode(audio_b64)
    except Exception as e:
        fail(f"Base64 decode failed: {e}")
        return False

    ok(f"Base64 decoded: {len(audio_b64)} chars -> {len(wav_data)} bytes")

    valid, desc = validate_wav(wav_data)
    if valid:
        ok(f"Valid WAV: {desc}")
    else:
        fail(f"Invalid WAV: {desc}")
        passed = False

    if save_wav and valid:
        with open(save_wav, "wb") as f:
            f.write(wav_data)
        info(f"Saved to {save_wav}")

    # ---- Phase 4: Compare with non-streaming ----
    print(f"\n{C_BOLD}Phase 4: Non-streaming comparison (same seed={seed}){C_RESET}")

    payload_nostream = json.dumps({
        "input": text,
        "voice": "alloy",
        "seed": seed,
        "stream": False,
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{server}/v1/audio/speech",
        data=payload_nostream,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp2:
            ref_wav = resp2.read()
    except Exception as e:
        fail(f"Non-streaming request failed: {e}")
        return False

    ok(f"Non-streaming response: {len(ref_wav)} bytes")

    if wav_data == ref_wav:
        ok("WAV is byte-identical to non-streaming output")
    else:
        fail(f"WAV differs from non-streaming output "
             f"(stream={len(wav_data)} vs nostream={len(ref_wav)} bytes)")
        passed = False

        # Diagnostic: compare headers and PCM
        if len(wav_data) >= 44 and len(ref_wav) >= 44:
            if wav_data[:44] == ref_wav[:44]:
                info("WAV headers match")
                # Find first differing byte
                min_len = min(len(wav_data), len(ref_wav))
                for k in range(44, min_len):
                    if wav_data[k] != ref_wav[k]:
                        info(f"First PCM difference at byte {k}")
                        break
            else:
                info("WAV headers differ")

    # ---- Summary ----
    print()
    if passed:
        print(f"{C_BGREEN}ALL CHECKS PASSED{C_RESET}  ({elapsed:.1f}s)")
    else:
        print(f"{C_BRED}SOME CHECKS FAILED{C_RESET}  ({elapsed:.1f}s)")

    return passed


def main() -> int:
    p = argparse.ArgumentParser(description="Test TTS SSE streaming")
    p.add_argument("--server", default="http://localhost:8090",
                    help="Server URL (default: http://localhost:8090)")
    p.add_argument("--text", default="Hello, world.",
                    help="Text to synthesize (default: 'Hello, world.')")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for determinism (default: 42)")
    p.add_argument("--verbose", action="store_true",
                    help="Print each SSE event")
    p.add_argument("--save-wav", metavar="PATH",
                    help="Save decoded WAV to file")
    args = p.parse_args()

    print(f"{C_BOLD}TTS Streaming Test{C_RESET}")
    print(f"Server: {args.server}")
    print(f"Text: \"{args.text}\"")
    print(f"Seed: {args.seed}")

    # Health check
    try:
        req = urllib.request.Request(f"{args.server}/health")
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception:
        print(f"\n{C_RED}Cannot connect to server at {args.server}{C_RESET}")
        return 2

    ok_result = run_test(
        server=args.server,
        text=args.text,
        seed=args.seed,
        verbose=args.verbose,
        save_wav=args.save_wav,
    )

    return 0 if ok_result else 1


if __name__ == "__main__":
    sys.exit(main())
