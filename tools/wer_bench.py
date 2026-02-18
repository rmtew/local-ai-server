#!/usr/bin/env python3
"""
Word Error Rate (WER) benchmark using LibriSpeech test sets.

Evaluates ASR transcription quality against ground-truth human transcripts.
Uses the server's HTTP API for transcription and ffmpeg for FLAC->WAV conversion.

Usage:
  # Run against test-clean (default)
  python tools/wer_bench.py

  # Run against test-other (harder speakers)
  python tools/wer_bench.py --split test-other

  # Limit number of utterances (useful for quick checks)
  python tools/wer_bench.py --limit 50

  # Custom server port
  python tools/wer_bench.py --port 8090

  # Show per-utterance details
  python tools/wer_bench.py --verbose

  # Use the CLI binary instead of HTTP server
  python tools/wer_bench.py --binary path/to/qwen_asr --model-dir path/to/model

Dataset: LibriSpeech (https://www.openslr.org/12/)
Expected location: DEPS_ROOT/datasets/librispeech/LibriSpeech/{test-clean,test-other}/
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---- ANSI colors ----

_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _sgr(code: str) -> str:
    return f"\033[{code}m" if _USE_COLOR else ""

C_RESET   = _sgr("0")
C_BOLD    = _sgr("1")
C_DIM     = _sgr("2")
C_RED     = _sgr("31")
C_GREEN   = _sgr("32")
C_YELLOW  = _sgr("33")
C_CYAN    = _sgr("36")
C_BRED    = _sgr("1;31")
C_BGREEN  = _sgr("1;32")
C_BYELLOW = _sgr("1;33")
C_BCYAN   = _sgr("1;36")
C_BWHITE  = _sgr("1;37")


# ---- WER computation ----

def normalize_for_wer(text: str) -> List[str]:
    """Normalize text for WER: uppercase, strip punctuation, split to words.

    LibriSpeech ground truth is already uppercase with no punctuation.
    Model output may have mixed case and punctuation.
    """
    out = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            out.append(ch.upper())
        else:
            out.append(" ")
    return "".join(out).split()


def word_error_rate(ref_words: List[str], hyp_words: List[str]) -> Tuple[int, int, int, int]:
    """Compute WER components using dynamic programming.

    Returns (substitutions, insertions, deletions, ref_len).
    WER = (S + I + D) / ref_len
    """
    n = len(ref_words)
    m = len(hyp_words)

    # dp[i][j] = (cost, S, I, D) for ref[:i] vs hyp[:j]
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)  # all deletions
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)  # all insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                # substitution
                c_sub = dp[i-1][j-1]
                cost_sub = c_sub[0] + 1
                # insertion (extra word in hyp)
                c_ins = dp[i][j-1]
                cost_ins = c_ins[0] + 1
                # deletion (missing word in hyp)
                c_del = dp[i-1][j]
                cost_del = c_del[0] + 1

                best = min(cost_sub, cost_ins, cost_del)
                if best == cost_sub:
                    dp[i][j] = (best, c_sub[1]+1, c_sub[2], c_sub[3])
                elif best == cost_ins:
                    dp[i][j] = (best, c_ins[1], c_ins[2]+1, c_ins[3])
                else:
                    dp[i][j] = (best, c_del[1], c_del[2], c_del[3]+1)

    _, s, i, d = dp[n][m]
    return s, i, d, n


# ---- LibriSpeech parsing ----

@dataclass
class Utterance:
    utterance_id: str   # e.g. "1089-134686-0000"
    flac_path: Path
    reference: str      # ground truth transcript (uppercase, no punct)


def load_librispeech_split(root: Path) -> List[Utterance]:
    """Load all utterances from a LibriSpeech split directory."""
    utterances = []
    for trans_file in sorted(root.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                flac = chapter_dir / f"{utt_id}.flac"
                if flac.exists():
                    utterances.append(Utterance(
                        utterance_id=utt_id,
                        flac_path=flac,
                        reference=text,
                    ))
    return utterances


# ---- Transcription backends ----

def flac_to_wav_bytes(flac_path: Path) -> bytes:
    """Convert FLAC to 16kHz mono s16le WAV using ffmpeg.

    Uses a temp file because piped WAV output has 0xFFFFFFFF size fields
    which our WAV parser can't handle.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(flac_path), "-ar", "16000", "-ac", "1",
             tmp],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {flac_path}: {result.stderr.decode()[:200]}")
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def transcribe_http(wav_bytes: bytes, port: int, filename: str = "audio.wav") -> str:
    """Transcribe via the server's HTTP multipart API."""
    import urllib.request
    import json

    boundary = "----WerBenchBoundary"
    body = bytearray()
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += wav_bytes
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"http://localhost:{port}/v1/audio/transcriptions",
        data=bytes(body),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data.get("text", "")


def transcribe_cli(flac_path: Path, binary: Path, model_dir: Path) -> str:
    """Transcribe via CLI binary (converts FLAC->WAV via temp file)."""
    wav_bytes = flac_to_wav_bytes(flac_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [str(binary), "-d", str(model_dir), "-i", tmp_path,
             "-S", "0", "--silent"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CLI failed: {result.stderr[:200]}")
        return result.stdout.strip()
    finally:
        os.unlink(tmp_path)


# ---- Main ----

@dataclass
class Results:
    total_sub: int = 0
    total_ins: int = 0
    total_del: int = 0
    total_ref_words: int = 0
    total_utterances: int = 0
    total_time: float = 0.0
    total_audio_duration: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def wer(self) -> float:
        if self.total_ref_words == 0:
            return 0.0
        return (self.total_sub + self.total_ins + self.total_del) / self.total_ref_words

    @property
    def total_errors(self) -> int:
        return self.total_sub + self.total_ins + self.total_del


def get_audio_duration_s(flac_path: Path) -> float:
    """Get duration of FLAC file in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(flac_path)],
        capture_output=True, text=True, timeout=10,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def fmt_time(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


def main() -> int:
    ap = argparse.ArgumentParser(description="WER benchmark using LibriSpeech")
    ap.add_argument("--split", default="test-clean",
                    choices=["test-clean", "test-other"],
                    help="LibriSpeech split to evaluate (default: test-clean)")
    ap.add_argument("--data-root", default=None,
                    help="Path to LibriSpeech root (default: auto-detect from DEPS_ROOT)")
    ap.add_argument("--port", type=int, default=8090,
                    help="Server port for HTTP transcription (default: 8090)")
    ap.add_argument("--binary", default=None,
                    help="Path to CLI binary (if set, uses CLI instead of HTTP)")
    ap.add_argument("--model-dir", default=None,
                    help="Model directory for CLI mode")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max utterances to evaluate (0 = all)")
    ap.add_argument("--verbose", action="store_true",
                    help="Show per-utterance results")
    ap.add_argument("--show-errors", action="store_true",
                    help="Show ref vs hyp for utterances with WER > 0")
    args = ap.parse_args()

    # Find dataset
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        deps = os.environ.get("DEPS_ROOT", "")
        if not deps:
            # Try common locations
            for candidate in [
                Path("C:/Data/R/git/claude-repos/deps"),
                Path.home() / "deps",
            ]:
                if (candidate / "datasets" / "librispeech").exists():
                    deps = str(candidate)
                    break
        if not deps:
            print("Cannot find LibriSpeech data. Set DEPS_ROOT or use --data-root.", file=sys.stderr)
            return 2
        data_root = Path(deps) / "datasets" / "librispeech" / "LibriSpeech"

    split_dir = data_root / args.split
    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}", file=sys.stderr)
        print(f"Download from https://www.openslr.org/12/ and extract to {data_root.parent}/", file=sys.stderr)
        return 2

    # Load utterances
    print(f"{C_BOLD}Loading LibriSpeech {args.split}...{C_RESET}")
    utterances = load_librispeech_split(split_dir)
    if not utterances:
        print(f"No utterances found in {split_dir}", file=sys.stderr)
        return 2

    if args.limit > 0:
        utterances = utterances[:args.limit]

    total = len(utterances)
    mode = "CLI" if args.binary else f"HTTP :{args.port}"
    print(f"{C_BOLD}{total} utterances, mode: {mode}{C_RESET}")
    print()

    # Validate mode
    if args.binary:
        binary = Path(args.binary).resolve()
        if not binary.exists():
            print(f"Binary not found: {binary}", file=sys.stderr)
            return 2
        if not args.model_dir:
            print("--model-dir required when using --binary", file=sys.stderr)
            return 2
        model_dir = Path(args.model_dir).resolve()
    else:
        binary = None
        model_dir = None
        # Quick health check
        import urllib.request
        try:
            urllib.request.urlopen(f"http://localhost:{args.port}/health", timeout=5)
        except Exception as e:
            print(f"Server not reachable at port {args.port}: {e}", file=sys.stderr)
            print("Start the server first, or use --binary for CLI mode.", file=sys.stderr)
            return 2

    results = Results()
    t_start = time.monotonic()

    for idx, utt in enumerate(utterances, 1):
        ref_words = normalize_for_wer(utt.reference)

        try:
            t0 = time.monotonic()
            if binary:
                hyp_text = transcribe_cli(utt.flac_path, binary, model_dir)
            else:
                wav_bytes = flac_to_wav_bytes(utt.flac_path)
                hyp_text = transcribe_http(wav_bytes, args.port, f"{utt.utterance_id}.wav")
            elapsed = time.monotonic() - t0
        except Exception as e:
            results.errors.append(f"{utt.utterance_id}: {e}")
            if args.verbose:
                print(f"{C_RED}[ERR  {idx}/{total}]{C_RESET} {utt.utterance_id}: {e}")
            continue

        hyp_words = normalize_for_wer(hyp_text)
        s, i, d, ref_len = word_error_rate(ref_words, hyp_words)
        utt_wer = (s + i + d) / max(1, ref_len)

        duration = get_audio_duration_s(utt.flac_path)
        results.total_sub += s
        results.total_ins += i
        results.total_del += d
        results.total_ref_words += ref_len
        results.total_utterances += 1
        results.total_time += elapsed
        results.total_audio_duration += duration

        if args.verbose:
            wer_color = C_GREEN if utt_wer == 0 else (C_YELLOW if utt_wer < 0.1 else C_RED)
            print(
                f"[{idx:4d}/{total}] {utt.utterance_id} | "
                f"WER {wer_color}{utt_wer:5.1%}{C_RESET} "
                f"(S={s} I={i} D={d} / {ref_len} words) | "
                f"{C_DIM}{elapsed:.1f}s{C_RESET}"
            )
            if args.show_errors and utt_wer > 0:
                print(f"         {C_GREEN}ref: {' '.join(ref_words[:30])}{'...' if len(ref_words) > 30 else ''}{C_RESET}")
                print(f"         {C_RED}hyp: {' '.join(hyp_words[:30])}{'...' if len(hyp_words) > 30 else ''}{C_RESET}")
        elif idx % 100 == 0 or idx == total:
            elapsed_total = time.monotonic() - t_start
            eta = (elapsed_total / idx) * (total - idx) if idx < total else 0
            print(
                f"\r{C_CYAN}[{idx}/{total}]{C_RESET} "
                f"WER so far: {C_BWHITE}{results.wer:.2%}{C_RESET} "
                f"({results.total_errors}/{results.total_ref_words}) "
                f"ETA: {fmt_time(eta)}  ",
                end="", flush=True,
            )

    if not args.verbose:
        print()  # newline after progress

    total_time = time.monotonic() - t_start
    print()
    print(f"{C_BOLD}{'=' * 60}{C_RESET}")
    print(f"{C_BOLD}LibriSpeech {args.split} — WER Results{C_RESET}")
    print(f"{C_BOLD}{'=' * 60}{C_RESET}")
    print()

    wer_pct = results.wer * 100
    wer_color = C_BGREEN if wer_pct < 3 else (C_BYELLOW if wer_pct < 8 else C_BRED)
    print(f"  WER:            {wer_color}{wer_pct:.2f}%{C_RESET}")
    print(f"  Substitutions:  {results.total_sub}")
    print(f"  Insertions:     {results.total_ins}")
    print(f"  Deletions:      {results.total_del}")
    print(f"  Total errors:   {results.total_errors} / {results.total_ref_words} words")
    print(f"  Utterances:     {results.total_utterances}")
    if results.errors:
        print(f"  Errors:         {C_RED}{len(results.errors)}{C_RESET}")
    print()
    print(f"  Audio duration: {fmt_time(results.total_audio_duration)}")
    print(f"  Wall time:      {fmt_time(total_time)}")
    if results.total_audio_duration > 0:
        rtf = total_time / results.total_audio_duration
        print(f"  RTF:            {rtf:.3f}x")
    print()

    if results.errors:
        print(f"{C_BRED}Transcription errors ({len(results.errors)}):{C_RESET}")
        for e in results.errors[:10]:
            print(f"  {e}")
        if len(results.errors) > 10:
            print(f"  ... and {len(results.errors) - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
