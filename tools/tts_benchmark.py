#!/usr/bin/env python3
"""
TTS performance benchmark across models, quantization modes, and voices.

Runs the TTS pipeline directly in-process via the native DLL (no HTTP server).
Measures pure synthesis time (decode + vocoder), step count, audio duration,
RTF (real-time factor), and ms/step across the full model x quant x voice matrix.

Uses the same 6 text prompts from tts_bench.c (4s-16s target duration), seed=42
for deterministic output. Reports median of N runs per case.

Requires: build.bat ttsdll (produces bin/tts_pipeline.dll)

Usage:
  python tools/tts_benchmark.py                        # Full matrix
  python tools/tts_benchmark.py --model 0.6b-cv        # Single model
  python tools/tts_benchmark.py --quant int8            # INT8 only
  python tools/tts_benchmark.py --quick                 # alloy only, 3 cases
  python tools/tts_benchmark.py --no-1.7b               # Skip 1.7B
  python tools/tts_benchmark.py --runs 5                # 5 iterations per case
  python tools/tts_benchmark.py --save                  # Append results to tts_benchmarks.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
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

SEED = 42
TEMPERATURE = 0.3

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

# ---- Benchmark cases (from tts_bench.c) ----

BENCH_CASES_FULL = [
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

BENCH_CASES_QUICK = BENCH_CASES_FULL[:3]

# ---- Benchmark data ----


class CaseTiming(NamedTuple):
    case_id: str
    steps: int
    audio_s: float
    total_ms: float  # median
    rtf: float
    ms_per_step: float


class ComboResult:
    def __init__(self, model_label: str, quant_label: str, voice: str):
        self.model_label = model_label
        self.quant_label = quant_label
        self.voice = voice
        self.vram_mb: Optional[int] = None
        self.cases: List[CaseTiming] = []

    @property
    def avg_ms_per_step(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.ms_per_step for c in self.cases) / len(self.cases)

    @property
    def avg_rtf(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.rtf for c in self.cases) / len(self.cases)


# ---- Helpers ----


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}m{s:.0f}s"


def median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


# ---- Core benchmark ----


def bench_combo(
    tts: TtsPipeline,
    model: ModelDef,
    quant_label: str,
    voice: str,
    cases: List[Tuple[str, str]],
    runs: int,
    warmup: int,
) -> ComboResult:
    """Benchmark one (model, quant, voice) combination."""
    result = ComboResult(model.label, quant_label, voice)

    # Warmup
    if warmup > 0:
        print(f"  {C_DIM}Warmup ({warmup} run{'s' if warmup > 1 else ''})...{C_RESET}",
              end="", flush=True)
        for _ in range(warmup):
            tts.synthesize(cases[0][1], voice=voice, seed=SEED)
        print(f" {C_DIM}done{C_RESET}")

    for case_id, text in cases:
        text_preview = text[:55] + "..." if len(text) > 55 else text
        print(f"  {case_id:>4s}  \"{text_preview}\"")

        timings_ms: List[float] = []
        steps = 0
        audio_s = 0.0

        for r in range(runs):
            try:
                sr = tts.synthesize(text, voice=voice, seed=SEED,
                                    temperature=TEMPERATURE)
            except RuntimeError as e:
                print(f"    {C_RED}run {r+1}: FAILED ({e}){C_RESET}")
                continue

            timings_ms.append(sr.elapsed_ms)
            steps = sr.n_steps
            audio_s = sr.duration_s

            print(f"    run {r+1}: {sr.elapsed_ms:7.0f} ms | {sr.n_steps:3d} steps | "
                  f"{sr.duration_s:5.1f}s audio | RTF {sr.elapsed_ms / (sr.duration_s * 1000):.3f}x"
                  if sr.duration_s > 0 else
                  f"    run {r+1}: {sr.elapsed_ms:7.0f} ms | 0 steps | 0.0s audio")

        if not timings_ms:
            continue

        med_ms = median(timings_ms)
        rtf = med_ms / (audio_s * 1000.0) if audio_s > 0 else 0.0
        ms_step = med_ms / steps if steps > 0 else 0.0

        result.cases.append(CaseTiming(
            case_id=case_id,
            steps=steps,
            audio_s=audio_s,
            total_ms=med_ms,
            rtf=rtf,
            ms_per_step=ms_step,
        ))

    return result


# ---- Markdown output ----


def format_markdown_section(
    results: List[ComboResult],
    runs: int,
    title: Optional[str] = None,
) -> str:
    """Format results as a markdown section for tts_benchmarks.md."""
    lines: List[str] = []
    today = date.today().isoformat()

    if title:
        lines.append(f"## {today} -- {title}")
    else:
        lines.append(f"## {today} -- In-Process Benchmark (tts_benchmark.py)")

    lines.append("")
    lines.append(f"**Method:** In-process DLL (no HTTP), seed={SEED}, "
                 f"median of {runs} runs")
    lines.append("")

    # Group by (model, quant)
    groups: Dict[Tuple[str, str], List[ComboResult]] = {}
    for r in results:
        key = (r.model_label, r.quant_label)
        groups.setdefault(key, []).append(r)

    for (model_label, quant_label), combo_results in groups.items():
        vram = combo_results[0].vram_mb
        vram_str = f", VRAM: {vram} MB" if vram else ""
        lines.append(f"### {model_label} / {quant_label}{vram_str}")
        lines.append("")

        for cr in combo_results:
            if len(combo_results) > 1:
                lines.append(f"**Voice: {cr.voice}**")
                lines.append("")

            lines.append("| Case | Steps | Audio | Total (median) | RTF | ms/step |")
            lines.append("|------|------:|------:|---------------:|----:|--------:|")

            for c in cr.cases:
                lines.append(
                    f"| {c.case_id} | {c.steps:>5d} | {c.audio_s:>5.1f}s | "
                    f"{c.total_ms:>10.0f} ms | {c.rtf:.2f}x | {c.ms_per_step:.1f} |"
                )
            lines.append("")

    # Cross-combo comparison
    if len(results) > 1:
        lines.append("### Summary")
        lines.append("")
        lines.append("| Model | Quant | Voice | Avg ms/step | Avg RTF | VRAM |")
        lines.append("|-------|-------|-------|------------:|--------:|-----:|")
        for cr in results:
            vram_str = f"{cr.vram_mb} MB" if cr.vram_mb else "N/A"
            lines.append(
                f"| {cr.model_label} | {cr.quant_label} | {cr.voice} | "
                f"{cr.avg_ms_per_step:.1f} | {cr.avg_rtf:.2f}x | {vram_str} |"
            )
        lines.append("")

    return "\n".join(lines)


def save_to_benchmarks_md(markdown: str) -> None:
    """Append a section to tools/tts_benchmarks.md."""
    script_dir = Path(__file__).resolve().parent
    md_path = script_dir / "tts_benchmarks.md"

    if md_path.exists():
        existing = md_path.read_text(encoding="utf-8")
        if not existing.endswith("\n"):
            existing += "\n"
        content = existing + "---\n\n" + markdown + "\n"
    else:
        content = markdown + "\n"

    md_path.write_text(content, encoding="utf-8")
    print(f"\n{C_GREEN}Results saved to {md_path}{C_RESET}")


# ---- Main ----


def main() -> int:
    p = argparse.ArgumentParser(
        description="TTS performance benchmark (in-process DLL, no HTTP)"
    )
    p.add_argument("--model", choices=[m.key for m in MODELS],
                   help="Benchmark only this model")
    p.add_argument("--quant", choices=list(QUANT_MODES.keys()),
                   help="Benchmark only this quant mode")
    p.add_argument("--quick", action="store_true",
                   help="Alloy only, first 3 cases")
    p.add_argument("--no-1.7b", dest="no_17b", action="store_true",
                   help="Skip 1.7B model")
    p.add_argument("--runs", type=int, default=3,
                   help="Iterations per case (default: 3)")
    p.add_argument("--warmup", type=int, default=1,
                   help="Warmup iterations (default: 1)")
    p.add_argument("--threads", type=int, default=4,
                   help="CPU threads for vocoder (default: 4)")
    p.add_argument("--save", action="store_true",
                   help="Append results to tools/tts_benchmarks.md")
    p.add_argument("--title", default=None,
                   help="Custom title for the markdown section")
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

    cases = BENCH_CASES_QUICK if args.quick else BENCH_CASES_FULL

    # Verify model dirs exist
    for m in models_to_test:
        path = os.path.join(TTS_MODELS_DIR, m.dirname)
        if not os.path.isdir(path):
            print(f"{C_RED}Model not found: {path}{C_RESET}")
            return 2

    # Header
    print(f"{C_BOLD}TTS In-Process Benchmark{C_RESET}")
    print(f"Models: {', '.join(m.label for m in models_to_test)}")
    print(f"Quant:  {', '.join(QUANT_MODES[q]['label'] for q in quant_keys)}")
    print(f"Cases:  {len(cases)} texts, {args.runs} runs each, {args.warmup} warmup")
    print(f"Threads: {args.threads}")
    if args.quick:
        print(f"Mode:   {C_YELLOW}quick{C_RESET} (alloy only, {len(cases)} cases)")
    print()

    all_results: List[ComboResult] = []
    t_total = time.monotonic()

    for model in models_to_test:
        tts_model_path = os.path.join(TTS_MODELS_DIR, model.dirname)

        for quant_key in quant_keys:
            quant = QUANT_MODES[quant_key]
            voices = model.voices if not args.quick else ["alloy"]

            combo_label = f"{model.label} / {quant['label']}"
            print(f"{C_BOLD}{'=' * 60}{C_RESET}")
            print(f"{C_BWHITE}{combo_label}{C_RESET}  "
                  f"({len(voices)} voice{'s' if len(voices) > 1 else ''}, "
                  f"{len(cases)} cases)")
            print(f"{C_BOLD}{'=' * 60}{C_RESET}")

            # Load pipeline
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
                print(f" {C_RED}FAILED: {e}{C_RESET}")
                continue

            load_ms = (time.monotonic() - t_load) * 1000.0
            vram = tts.get_vram_mb()
            vram_str = f", {vram} MB VRAM" if vram else ""
            print(f" {C_GREEN}ready{C_RESET} ({load_ms:.0f} ms{vram_str})")

            # Benchmark each voice
            t_combo = time.monotonic()
            for voice in voices:
                print(f"\n  {C_BCYAN}[{voice}]{C_RESET}")
                cr = bench_combo(tts, model, quant["label"], voice,
                                 cases, args.runs, args.warmup)
                cr.vram_mb = vram
                all_results.append(cr)

            elapsed = time.monotonic() - t_combo
            print(f"\n  Elapsed: {fmt_time(elapsed)}")

            tts.close()
            print()

    # ---- Summary ----
    elapsed_total = time.monotonic() - t_total

    if not all_results:
        print(f"{C_RED}No results collected{C_RESET}")
        return 1

    print(f"{C_BOLD}{'=' * 78}{C_RESET}")
    print(f"{C_BOLD}Summary{C_RESET}  ({fmt_time(elapsed_total)} total)")
    print(f"{C_BOLD}{'=' * 78}{C_RESET}")

    # Per-combo case table
    for cr in all_results:
        vram_str = f" ({cr.vram_mb} MB)" if cr.vram_mb else ""
        print(f"\n{C_BWHITE}{cr.model_label} / {cr.quant_label} / {cr.voice}{vram_str}{C_RESET}")
        if cr.cases:
            print(f"  {'Case':>4s}  {'Steps':>5s}  {'Audio':>5s}  "
                  f"{'Total':>8s}  {'RTF':>6s}  {'ms/step':>7s}")
            print(f"  {'----':>4s}  {'-----':>5s}  {'-----':>5s}  "
                  f"{'--------':>8s}  {'------':>6s}  {'-------':>7s}")
            for c in cr.cases:
                print(f"  {c.case_id:>4s}  {c.steps:>5d}  {c.audio_s:>5.1f}s  "
                      f"{c.total_ms:>7.0f}ms  {c.rtf:>5.2f}x  {c.ms_per_step:>6.1f}")
        else:
            print(f"  {C_RED}No results{C_RESET}")

    # Cross-combo comparison
    if len(all_results) > 1:
        print(f"\n{C_BOLD}Cross-combo comparison:{C_RESET}")
        print(f"  {'Model':<12s} {'Quant':<6s} {'Voice':<8s} "
              f"{'Avg ms/step':>11s} {'Avg RTF':>8s} {'VRAM':>6s}")
        print(f"  {'-----':<12s} {'-----':<6s} {'-----':<8s} "
              f"{'-----------':>11s} {'-------':>8s} {'----':>6s}")
        for cr in all_results:
            vram_str = f"{cr.vram_mb:>4d}MB" if cr.vram_mb else "   N/A"
            if cr.cases:
                print(f"  {cr.model_label:<12s} {cr.quant_label:<6s} {cr.voice:<8s} "
                      f"{cr.avg_ms_per_step:>10.1f}  {cr.avg_rtf:>7.2f}x {vram_str}")

    # Save to markdown
    if args.save:
        md = format_markdown_section(all_results, args.runs, args.title)
        save_to_benchmarks_md(md)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
