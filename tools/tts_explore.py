#!/usr/bin/env python3
"""
TTS corner-case explorer — sweeps seed x text x voice x language combos to
discover failures and anomalies.

Runs the TTS pipeline directly in-process via the native DLL (no HTTP server).

Anomaly detection (no reference WAVs needed):
  - Max-steps hit: step count = max_steps (200) — likely runaway generation
  - Silence: RMS <= 0.001
  - Too short: audio < 0.3s for non-trivial text
  - Too long: audio > expected_duration x 2.5 (heuristic: ~0.08s/word)

Results feed back into tts_validate.py as specific regression cases.

Requires: build.bat ttsdll (produces bin/tts_pipeline.dll)

Usage:
  python tools/tts_explore.py                          # Full matrix, seeds 0-9
  python tools/tts_explore.py --model 0.6b-cv          # Single model
  python tools/tts_explore.py --quant fp16             # FP16 only
  python tools/tts_explore.py --seeds 0-49             # Wider seed sweep
  python tools/tts_explore.py --no-1.7b                # Skip 1.7B
  python tools/tts_explore.py --json                   # Machine-readable output
  python tools/tts_explore.py --quick                  # 3 texts, 5 seeds, alloy only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from tts_pipeline_ffi import TtsPipeline

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

TEMPERATURE = 0.3
MAX_STEPS_DEFAULT = 200

# ---- Model definitions ----

DEPS_ROOT = os.environ.get("DEPS_ROOT", "C:/Data/R/git/claude-repos/deps")
TTS_MODELS_DIR = os.path.join(DEPS_ROOT, "models", "tts")


class ModelDef(NamedTuple):
    key: str
    label: str
    dirname: str
    voices: List[str]
    is_1_7b: bool


MODELS: List[ModelDef] = [
    ModelDef("0.6b-cv", "0.6B-CV", "qwen3-tts-12hz-0.6b-customvoice",
             ["alloy", "serena", "ryan", "aiden"], False),
    ModelDef("0.6b-base", "0.6B-Base", "qwen3-tts-12hz-0.6b-base",
             ["alloy"], False),
    ModelDef("1.7b-base", "1.7B-Base", "qwen3-tts-12hz-1.7b-base",
             ["alloy"], True),
]

QUANT_MODES = {
    "f32":  {"label": "F32",  "fp16": False, "int8": False},
    "fp16": {"label": "FP16", "fp16": True,  "int8": False},
    "int8": {"label": "INT8", "fp16": False, "int8": True},
}

# ---- Explore text corpus ----

EXPLORE_TEXTS_FULL: List[Tuple[str, str]] = [
    # English — variety of lengths and content
    ("en_short",    "Hello, world."),
    ("en_medium",   "The quick brown fox jumps over the lazy dog near the riverbank."),
    ("en_long",
     "In the beginning, the universe was created. This has made a lot of "
     "people very angry and been widely regarded as a bad move. Many were "
     "increasingly of the opinion that they had all made a big mistake "
     "in coming down from the trees in the first place."),
    ("en_numbers",  "Call 1-800-555-0199 or visit 42 Oak Street, Suite 3B."),
    ("en_punct",    "Wait... really? Yes! Absolutely \u2014 no doubt about it."),
    ("en_names",    "Dr. Elizabeth Warren met with CEO Satya Nadella at Microsoft."),
    # Chinese
    ("zh_short",    "\u4f60\u597d\u4e16\u754c"),
    ("zh_medium",   "\u4eca\u5929\u5929\u6c14\u771f\u597d\uff0c\u6211\u4eec\u53bb\u516c\u56ed\u6563\u6b65\u5427\u3002\u82b1\u513f\u90fd\u5f00\u4e86\uff0c\u975e\u5e38\u6f02\u4eae\u3002"),
    ("zh_en_mix",   "\u6211\u5728Google\u5de5\u4f5c\uff0c\u6bcf\u5929\u7528Python\u5199\u4ee3\u7801\u3002"),
    # Mixed / code-switching
    ("mixed_1",     "The caf\u00e9 serves excellent cr\u00e8me br\u00fbl\u00e9e for just $8.50."),
]

EXPLORE_TEXTS_QUICK: List[Tuple[str, str]] = [
    EXPLORE_TEXTS_FULL[0],  # en_short
    EXPLORE_TEXTS_FULL[1],  # en_medium
    EXPLORE_TEXTS_FULL[2],  # en_long
]

# ---- WAV helpers ----


def read_wav_samples(data: bytes) -> Optional[List[float]]:
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


# ---- Text normalization ----


def word_count(text: str) -> int:
    """Rough word count for duration heuristic."""
    cjk = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    ascii_words = len(re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf]', '', text).split())
    return cjk + ascii_words


# ---- Anomaly detection ----


class Anomaly(NamedTuple):
    kind: str       # "max-steps", "silence", "too-short", "too-long", "error"
    detail: str


def detect_anomalies(
    wav: bytes,
    text: str,
    n_steps: int,
    max_steps: int = MAX_STEPS_DEFAULT,
) -> List[Anomaly]:
    """Check a TTS output for anomalies. No reference WAV needed."""
    anomalies: List[Anomaly] = []

    samples = read_wav_samples(wav)
    if samples is None:
        anomalies.append(Anomaly("error", "invalid WAV"))
        return anomalies

    duration = len(samples) / 24000.0
    rms = math.sqrt(sum(x * x for x in samples) / len(samples)) if samples else 0.0

    # Max-steps hit
    if n_steps >= max_steps:
        anomalies.append(Anomaly("max-steps",
                                 f"{n_steps} steps, {duration:.1f}s"))

    # Silence
    if rms <= 0.001:
        anomalies.append(Anomaly("silence", f"RMS={rms:.6f}"))

    # Too short (for non-trivial text)
    wc = word_count(text)
    if wc >= 3 and duration < 0.3:
        anomalies.append(Anomaly("too-short",
                                 f"{duration:.2f}s for {wc} words"))

    # Too long (heuristic: ~0.08s per word, with 2.5x tolerance)
    if wc > 0:
        expected_dur = wc * 0.08
        max_dur = max(expected_dur * 2.5, 3.0)
        if duration > max_dur:
            anomalies.append(Anomaly("too-long",
                                     f"{duration:.1f}s (expected <{max_dur:.1f}s "
                                     f"for {wc} words)"))

    return anomalies


# ---- Explore results ----


class ExploreHit(NamedTuple):
    text_id: str
    text: str
    seed: int
    voice: str
    anomalies: List[Anomaly]


class VoiceStats:
    def __init__(self, voice: str):
        self.voice = voice
        self.total = 0
        self.hits: List[ExploreHit] = []

    @property
    def anomaly_count(self) -> int:
        return len(self.hits)

    @property
    def rate(self) -> float:
        return self.anomaly_count / self.total if self.total > 0 else 0.0


class ComboStats:
    def __init__(self, model_label: str, quant_label: str):
        self.model_label = model_label
        self.quant_label = quant_label
        self.voice_stats: Dict[str, VoiceStats] = {}

    def get_voice(self, voice: str) -> VoiceStats:
        if voice not in self.voice_stats:
            self.voice_stats[voice] = VoiceStats(voice)
        return self.voice_stats[voice]

    @property
    def total_requests(self) -> int:
        return sum(vs.total for vs in self.voice_stats.values())

    @property
    def total_anomalies(self) -> int:
        return sum(vs.anomaly_count for vs in self.voice_stats.values())

    @property
    def worst_anomaly(self) -> str:
        worst_kinds: Dict[str, int] = {}
        for vs in self.voice_stats.values():
            for hit in vs.hits:
                for a in hit.anomalies:
                    worst_kinds[a.kind] = worst_kinds.get(a.kind, 0) + 1
        if not worst_kinds:
            return "-"
        top = max(worst_kinds, key=worst_kinds.get)
        return f"{top} (x{worst_kinds[top]})"


# ---- Helpers ----


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}m{s:.0f}s"


def parse_seed_range(s: str) -> List[int]:
    """Parse '0-9' or '0-49' or '5' into list of ints."""
    if "-" in s:
        parts = s.split("-", 1)
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(s)]


# ---- Core explore ----


def explore_combo(
    tts: TtsPipeline,
    model: ModelDef,
    quant_label: str,
    voices: List[str],
    texts: List[Tuple[str, str]],
    seeds: List[int],
    max_steps: int,
) -> ComboStats:
    """Explore one (model, quant) combination across voices x texts x seeds."""
    stats = ComboStats(model.label, quant_label)

    total_per_voice = len(texts) * len(seeds)

    for voice in voices:
        vs = stats.get_voice(voice)
        vs.total = total_per_voice

        print(f"  {C_BCYAN}[{voice}]{C_RESET}  ", end="", flush=True)

        for text_id, text in texts:
            for seed in seeds:
                try:
                    sr = tts.synthesize(text, voice=voice, seed=seed,
                                        temperature=TEMPERATURE)
                except RuntimeError:
                    hit = ExploreHit(text_id, text, seed, voice,
                                     [Anomaly("error", "synthesis failed")])
                    vs.hits.append(hit)
                    print(f"{C_RED}E{C_RESET}", end="", flush=True)
                    continue

                anomalies = detect_anomalies(sr.wav_data, text, sr.n_steps,
                                              max_steps)

                if anomalies:
                    hit = ExploreHit(text_id, text, seed, voice, anomalies)
                    vs.hits.append(hit)
                    print(f"{C_RED}X{C_RESET}", end="", flush=True)
                else:
                    print(".", end="", flush=True)

        anomaly_count = vs.anomaly_count
        if anomaly_count > 0:
            print(f"  {C_RED}{anomaly_count} anomalies{C_RESET}")
        else:
            print(f"  {C_GREEN}0 anomalies{C_RESET}")

        # Print details of anomalies
        for hit in vs.hits:
            text_preview = hit.text[:40] + "..." if len(hit.text) > 40 else hit.text
            kinds = ", ".join(f"{a.kind}: {a.detail}" for a in hit.anomalies)
            print(f"    seed={hit.seed} [{hit.text_id}] \"{text_preview}\": {kinds}")

    return stats


# ---- JSON output ----


def stats_to_json(all_stats: List[ComboStats]) -> str:
    """Convert all results to JSON."""
    output = []
    for cs in all_stats:
        combo = {
            "model": cs.model_label,
            "quant": cs.quant_label,
            "total_requests": cs.total_requests,
            "total_anomalies": cs.total_anomalies,
            "voices": {},
        }
        for voice, vs in cs.voice_stats.items():
            voice_data = {
                "total": vs.total,
                "anomaly_count": vs.anomaly_count,
                "rate": round(vs.rate, 4),
                "hits": [],
            }
            for hit in vs.hits:
                voice_data["hits"].append({
                    "text_id": hit.text_id,
                    "seed": hit.seed,
                    "anomalies": [{"kind": a.kind, "detail": a.detail}
                                  for a in hit.anomalies],
                })
            combo["voices"][voice] = voice_data
        output.append(combo)
    return json.dumps(output, indent=2)


# ---- Main ----


def main() -> int:
    p = argparse.ArgumentParser(
        description="TTS corner-case explorer (in-process DLL, no HTTP)"
    )
    p.add_argument("--model", choices=[m.key for m in MODELS],
                   help="Explore only this model")
    p.add_argument("--quant", choices=list(QUANT_MODES.keys()),
                   help="Explore only this quant mode")
    p.add_argument("--seeds", default="0-9",
                   help="Seed range (default: 0-9)")
    p.add_argument("--quick", action="store_true",
                   help="3 texts, 5 seeds, alloy only")
    p.add_argument("--no-1.7b", dest="no_17b", action="store_true",
                   help="Skip 1.7B model")
    p.add_argument("--threads", type=int, default=4,
                   help="CPU threads for vocoder (default: 4)")
    p.add_argument("--json", dest="json_output", action="store_true",
                   help="Machine-readable JSON output")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT,
                   help=f"Max decode steps for anomaly detection (default: {MAX_STEPS_DEFAULT})")
    p.add_argument("--verbose", action="store_true",
                   help="Show DLL init output")

    args = p.parse_args()

    # Build matrix
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

    # Seeds
    if args.quick:
        seeds = list(range(5))
        texts = EXPLORE_TEXTS_QUICK
    else:
        seeds = parse_seed_range(args.seeds)
        texts = EXPLORE_TEXTS_FULL

    # Verify model dirs
    for m in models_to_test:
        path = os.path.join(TTS_MODELS_DIR, m.dirname)
        if not os.path.isdir(path):
            print(f"{C_RED}Model not found: {path}{C_RESET}")
            return 2

    # Header
    if not args.json_output:
        print(f"{C_BOLD}TTS Corner-Case Explorer{C_RESET}")
        print(f"Models: {', '.join(m.label for m in models_to_test)}")
        print(f"Quant:  {', '.join(QUANT_MODES[q]['label'] for q in quant_keys)}")
        print(f"Texts:  {len(texts)}, Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)})")
        print(f"Threads: {args.threads}")
        if args.quick:
            print(f"Mode:   {C_YELLOW}quick{C_RESET}")
        print()

    all_stats: List[ComboStats] = []
    t_total = time.monotonic()

    for model in models_to_test:
        tts_model_path = os.path.join(TTS_MODELS_DIR, model.dirname)

        for quant_key in quant_keys:
            quant = QUANT_MODES[quant_key]
            voices = model.voices if not args.quick else ["alloy"]

            combo_label = f"{model.label} / {quant['label']}"
            total_requests = len(texts) * len(seeds) * len(voices)

            if not args.json_output:
                print(f"{C_BOLD}{'=' * 60}{C_RESET}")
                print(f"{C_BWHITE}{combo_label}{C_RESET}  "
                      f"({len(texts)} texts x {len(seeds)} seeds x "
                      f"{len(voices)} voices = {total_requests} requests)")
                print(f"{C_BOLD}{'=' * 60}{C_RESET}")

            # Load pipeline
            if not args.json_output:
                print(f"  Loading model...", end="", flush=True)

            t_load = time.monotonic()
            try:
                tts = TtsPipeline(
                    tts_model_path,
                    fp16=quant["fp16"],
                    int8=quant["int8"],
                    verbose=args.verbose,
                    threads=args.threads,
                )
            except Exception as e:
                if not args.json_output:
                    print(f" {C_RED}FAILED: {e}{C_RESET}")
                continue

            load_ms = (time.monotonic() - t_load) * 1000.0
            if not args.json_output:
                print(f" {C_GREEN}ready{C_RESET} ({load_ms:.0f} ms)")

            # Explore
            t_combo = time.monotonic()
            cs = explore_combo(
                tts, model, quant["label"], voices,
                texts, seeds, args.max_steps,
            )
            all_stats.append(cs)

            tts.close()

            if not args.json_output:
                elapsed = time.monotonic() - t_combo
                print(f"\n  Elapsed: {fmt_time(elapsed)}")
                print()

    # ---- Output ----
    elapsed_total = time.monotonic() - t_total

    if args.json_output:
        print(stats_to_json(all_stats))
        return 0

    # Summary
    total_requests = sum(cs.total_requests for cs in all_stats)
    total_anomalies = sum(cs.total_anomalies for cs in all_stats)

    print(f"{C_BOLD}{'=' * 78}{C_RESET}")
    print(f"{C_BOLD}Anomaly Summary{C_RESET}  "
          f"({total_requests} requests, {fmt_time(elapsed_total)} total)")
    print(f"{C_BOLD}{'=' * 78}{C_RESET}")

    if total_anomalies == 0:
        print(f"\n  {C_BGREEN}No anomalies found!{C_RESET}\n")
        return 0

    print(f"\n  {'Model':<12s} {'Quant':<6s} {'Voice':<8s} "
          f"{'Anomalies':>9s} {'Rate':>6s}   {'Worst'}")
    print(f"  {'-----':<12s} {'-----':<6s} {'-----':<8s} "
          f"{'---------':>9s} {'----':>6s}   {'-----'}")

    for cs in all_stats:
        for voice, vs in cs.voice_stats.items():
            if vs.anomaly_count == 0:
                rate_color = C_GREEN
            elif vs.rate < 0.05:
                rate_color = C_YELLOW
            else:
                rate_color = C_RED

            kind_counts: Dict[str, int] = {}
            for hit in vs.hits:
                for a in hit.anomalies:
                    kind_counts[a.kind] = kind_counts.get(a.kind, 0) + 1
            worst = ""
            if kind_counts:
                top_kind = max(kind_counts, key=kind_counts.get)
                seeds_hit = sorted(set(h.seed for h in vs.hits
                                       if any(a.kind == top_kind
                                              for a in h.anomalies)))
                seed_str = ",".join(str(s) for s in seeds_hit[:5])
                if len(seeds_hit) > 5:
                    seed_str += "..."
                worst = f"{top_kind} (seed={seed_str})"

            print(f"  {cs.model_label:<12s} {cs.quant_label:<6s} {voice:<8s} "
                  f"{vs.anomaly_count:>3d}/{vs.total:<5d} "
                  f"{rate_color}{vs.rate:>5.1%}{C_RESET}   {worst}")

    # Detailed anomaly list
    print(f"\n{C_BOLD}Details:{C_RESET}")
    for cs in all_stats:
        for voice, vs in cs.voice_stats.items():
            for hit in vs.hits:
                text_preview = hit.text[:35] + "..." if len(hit.text) > 35 else hit.text
                kinds = ", ".join(a.kind for a in hit.anomalies)
                print(f"  {C_RED}{kinds}{C_RESET}  "
                      f"{cs.model_label}/{cs.quant_label} [{voice}] "
                      f"seed={hit.seed} [{hit.text_id}] \"{text_preview}\"")

    print()
    return 1 if total_anomalies > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
