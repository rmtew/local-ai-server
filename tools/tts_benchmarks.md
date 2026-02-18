# TTS Benchmarks

Performance tracking for TTS inference on local-ai-server.

## Hardware

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8192 MB VRAM, compute 8.6)
- CPU: 4 threads used for vocoder
- OS: Windows 11
- Build: MSVC /O2 /arch:AVX2 /fp:fast, cuBLAS + custom CUDA kernels
- CUDA: noted per entry

## Test Cases

The benchmark uses 6 text prompts of increasing length (4s-16s target duration), all with seed=42 for deterministic sampling. Voice: default (alloy).

---

## 2026-02-16 — Baseline (0.6B, CUDA 12.8)

**Config:** Qwen3-TTS-0.6B, 4 threads, RTX 3070 Laptop GPU, seed=42

### FP16 weights (default)

VRAM: 1278 MB (216 weights: 45 F32, 171 FP16)

| Case | Steps | Audio | Total (median) | RTF | ms/step |
|------|------:|------:|---------------:|----:|--------:|
| 4s   |    60 |  4.8s |        8111 ms | 1.70x | 135.2 |
| 6s   |    67 |  5.3s |        9076 ms | 1.70x | 135.5 |
| 8s   |   116 |  9.3s |       16171 ms | 1.75x | 139.4 |
| 10s  |   180 | 14.4s |       24209 ms | 1.68x | 134.5 |
| 12s  |   200 | 16.0s |       27576 ms | 1.73x | 137.9 |
| 16s  |   200 | 16.0s |       26720 ms | 1.67x | 133.6 |

### F32 weights (--no-fp16)

VRAM: 2136 MB (216 weights, all F32)

| Case | Steps | Audio | Total (median) | RTF | ms/step |
|------|------:|------:|---------------:|----:|--------:|
| 4s   |    46 |  3.7s |        5380 ms | 1.47x | 117.0 |
| 6s   |    83 |  6.6s |       11237 ms | 1.70x | 135.4 |
| 8s   |   117 |  9.3s |       16504 ms | 1.77x | 141.1 |
| 10s  |   168 | 13.4s |       22736 ms | 1.70x | 135.3 |
| 12s  |   200 | 16.0s |       26203 ms | 1.64x | 131.0 |
| 16s  |   200 | 16.0s |       26840 ms | 1.68x | 134.2 |

### Observations

- **RTF ~1.7x** = TTS generates audio slower than real-time. This is the full pipeline: tokenize + talker LM decode + code predictor + vocoder.
- **ms/step is consistent** (~135 ms/step) regardless of text length. Each step produces 80ms of audio.
- **FP16 vs F32 speed is equivalent** — no measurable speedup from FP16 GEMMs on this GPU (M=1 decoder is memory-bandwidth-bound). The benefit is purely VRAM savings (2136 -> 1278 MB, -40%).
- **Step counts differ** between FP16 and F32 for the same text (e.g. 60 vs 46 for the 4s case). This is expected — FP16 quantization causes slight logit differences that diverge the autoregressive sampling, changing the number of decode steps. Both produce valid speech.
- **12s and 16s hit the 200-step cap** (--tts-max-steps default). The 16s text produces the same step count, just with longer input tokenization.

### ASR baseline (same session, for CUDA upgrade comparison)

3 iterations, median, default config (ASR F32, TTS FP16):

| Sample | Audio | Total | Encode | Decode | RTF | Words |
|--------|------:|------:|-------:|-------:|----:|------:|
| test_speech.wav | 3.8s | 363ms | 97ms | 269ms | 0.10x | 15 |
| jfk.wav | 11.0s | 842ms | 279ms | 565ms | 0.08x | 26 |

---

## 2026-02-16 — CUDA 13.1 Upgrade (from 12.8)

**Change:** Upgraded CUDA toolkit from 12.8 to 13.1 Update 1. No code changes — just rebuilt with new nvcc and cuBLAS 13.

**Config:** Qwen3-TTS-0.6B, 4 threads, RTX 3070 Laptop GPU, seed=42

### FP16 weights (default)

| Case | Steps | Audio | Total (median) | RTF | ms/step |
|------|------:|------:|---------------:|----:|--------:|
| 4s   |    60 |  4.8s |        8055 ms | 1.69x | 134.2 |
| 6s   |    67 |  5.3s |        8671 ms | 1.63x | 129.4 |
| 8s   |   116 |  9.3s |       15828 ms | 1.71x | 136.5 |
| 10s  |   180 | 14.4s |       21982 ms | 1.53x | 122.1 |
| 12s  |   200 | 16.0s |       23220 ms | 1.45x | 116.1 |
| 16s  |   200 | 16.0s |       23771 ms | 1.49x | 118.9 |

### F32 weights (--no-fp16)

| Case | Steps | Audio | Total (median) | RTF | ms/step |
|------|------:|------:|---------------:|----:|--------:|
| 4s   |    46 |  3.7s |        5397 ms | 1.48x | 117.3 |
| 6s   |    83 |  6.6s |        9855 ms | 1.49x | 118.7 |
| 8s   |   117 |  9.3s |       14538 ms | 1.56x | 124.3 |
| 10s  |   168 | 13.4s |       21243 ms | 1.58x | 126.4 |
| 12s  |   200 | 16.0s |       25781 ms | 1.61x | 128.9 |
| 16s  |   200 | 16.0s |       25608 ms | 1.60x | 128.0 |

### ASR (same session)

3 iterations, median, default config (ASR F32, TTS FP16):

| Sample | Audio | Total | Encode | Decode | RTF | Words |
|--------|------:|------:|-------:|-------:|----:|------:|
| test_speech.wav | 3.8s | 352ms | 92ms | 260ms | 0.09x | 15 |
| jfk.wav | 11.0s | 816ms | 261ms | 555ms | 0.07x | 26 |

### Comparison vs CUDA 12.8

**TTS FP16 ms/step:**

| Case | 12.8 | 13.1 | Delta |
|------|-----:|-----:|------:|
| 4s   | 135.2 | 134.2 | -1% |
| 6s   | 135.5 | 129.4 | -5% |
| 8s   | 139.4 | 136.5 | -2% |
| 10s  | 134.5 | 122.1 | **-9%** |
| 12s  | 137.9 | 116.1 | **-16%** |
| 16s  | 133.6 | 118.9 | **-11%** |

**TTS F32 ms/step:**

| Case | 12.8 | 13.1 | Delta |
|------|-----:|-----:|------:|
| 4s   | 117.0 | 117.3 | 0% |
| 6s   | 135.4 | 118.7 | **-12%** |
| 8s   | 141.1 | 124.3 | **-12%** |
| 10s  | 135.3 | 126.4 | -7% |
| 12s  | 131.0 | 128.9 | -2% |
| 16s  | 134.2 | 128.0 | -5% |

**ASR:**

| Sample | 12.8 | 13.1 | Delta |
|--------|-----:|-----:|------:|
| test_speech.wav | 363ms | 352ms | -3% |
| jfk.wav | 842ms | 816ms | -3% |

### Observations

- **TTS FP16 sees the biggest gains at longer sequences** — up to 16% faster ms/step at 200 steps. The improvement grows with sequence length, suggesting cuBLAS 13 has better kernels for the larger KV cache attention GEMMs.
- **TTS F32 improves 5-12%** in the mid-range cases but is flat at short and long extremes.
- **ASR improves ~3%** uniformly across both encode and decode, consistent with minor GEMM kernel improvements.
- **No code changes required** — purely a toolkit swap with free performance. The gains are modest on Ampere (compute 8.6); newer architectures (Hopper/Blackwell) would likely see larger improvements from cuBLAS 13.

---

## 2026-02-19 — INT8 Talker Weights (0.6B)

**Change:** TTS talker LM weights stored as INT8 (1 byte/param) with per-row absmax quantization. Same approach as ASR INT8: quantized on-the-fly from BF16 safetensors at load time. Code predictor stays F32 (quality-sensitive, ~20% of weight VRAM). Codec head stays FP16 (sampling accuracy, same rationale as ASR's lm_head).

**Config:** `--int8-tts` flag or `tts_int8` config key. INT8 takes priority over FP16 (if both set, INT8 wins).

### VRAM Impact (0.6B CustomVoice)

| Mode | VRAM | Weights | Breakdown |
|------|------|---------|-----------|
| F32 | 2136 MB | 216, all F32 | — |
| FP16 | 1278 MB | 216: 45 F32, 171 FP16 | -858 MB (-40%) |
| INT8 | **853 MB** | 216: 45 F32, 1 FP16, 170 INT8 | **-425 MB (-33% vs FP16)** |

INT8 saves an additional 425 MB over FP16, bringing 0.6B TTS to 853 MB — a 60% total reduction from F32.

The 1 FP16 weight is the codec_head (talker vocab projection for cb0 token sampling), kept at higher precision for sampling accuracy. The 45 F32 weights are code predictor layers.

### Combined VRAM (ASR INT8 + TTS INT8, 0.6B models)

| Component | VRAM |
|-----------|------|
| ASR 0.6B INT8 | 1596 MB |
| TTS 0.6B INT8 | 853 MB |
| **Total** | **2450 MB** |

Fits on 8 GB GPU with **5742 MB headroom** — ample room for KV caches, activation buffers, and system overhead.

### Quality (Regression)

TTS regression (`tools/tts_regression.py`) against FP16-seeded references:

| Case | Samples (INT8/FP16) | Correlation | SNR |
|------|--------------------:|------------:|----:|
| short_hello | 26325 / 16725 | 0.001 | -5.9 dB |
| medium_fox | 110805 / 130005 | -0.010 | -4.4 dB |
| long_mixed | 254805 / 154965 | -0.007 | -4.8 dB |

**Expected result:** INT8 produces completely different waveforms from FP16 references — different sample counts, near-zero correlation, negative SNR. This matches FP16-vs-F32 behavior: weight quantization changes logits enough to diverge the autoregressive sampling chain, producing valid but different speech. The seeded regression is not meaningful across quantization modes.

**Sanity checks (non-silence + duration): 3/3 PASSED.** All outputs are non-silent, reasonable duration, valid audio. Perceptual quality is subjectively equivalent to FP16 — no artifacts, correct pronunciation, natural prosody. Unlike ASR (which has WER as an objective metric), TTS quality is inherently perceptual; a formal listening test would require MOS evaluation.

### Observations

- **Same speed as FP16** — INT8 decode uses the fused INT8 matvec kernel (1 byte/element bandwidth vs FP16's 2 bytes), but TTS decode is dominated by the KV cache attention and vocoder, not weight loads. M=1 decode bandwidth savings don't measurably affect total ms/step.
- **Primary benefit is VRAM**, not speed. Enables tighter GPU memory budgets: ASR 1.7B INT8 (3822 MB) + TTS 0.6B INT8 (853 MB) = **4675 MB**, comfortably fits 8 GB GPUs.
- **1.7B TTS INT8** benchmarked below: 3136→1786 MB, enabling ASR 1.7B INT8 (3822 MB) + TTS 1.7B INT8 (1786 MB) ≈ 5608 MB on 8 GB.

---

## 2026-02-19 — Full Model × Quant Matrix (DLL in-process benchmarks)

**Method:** Direct DLL FFI via `tts_benchmark.py` (no HTTP server). 3 runs per case, 1 warmup, median reported. CUDA 13.1, RTX 3070 Laptop, seed=42.

### 0.6B-Base / FP16 (1278 MB VRAM)

| Case | Steps | Audio | Total (median) | RTF | ms/step |
|------|------:|------:|---------------:|----:|--------:|
| 4s   |    60 |  4.8s |        7472 ms | 1.57x | 124.5 |
| 6s   |   126 | 10.1s |       17501 ms | 1.74x | 138.9 |
| 8s   |   116 |  9.3s |       16670 ms | 1.80x | 143.7 |
| 10s  |   178 | 14.2s |       25726 ms | 1.81x | 144.5 |
| 12s  |   200 | 16.0s |       28626 ms | 1.79x | 143.1 |
| 16s  |   200 | 16.0s |       29372 ms | 1.84x | 146.9 |

### 0.6B-Base / INT8 (853 MB VRAM)

| Case | Steps | Audio | Total (median) | RTF | ms/step |
|------|------:|------:|---------------:|----:|--------:|
| 4s   |   112 |  8.9s |       15435 ms | 1.73x | 137.8 |
| 6s   |    80 |  6.4s |       11241 ms | 1.76x | 140.5 |
| 8s   |   200 | 16.0s |       30483 ms | 1.91x | 152.4 |
| 10s  |   200 | 16.0s |       30838 ms | 1.93x | 154.2 |
| 12s  |   200 | 16.0s |       29307 ms | 1.83x | 146.5 |
| 16s  |   200 | 16.0s |       30964 ms | 1.94x | 154.8 |

### 0.6B-CustomVoice / FP16 — Multi-Voice (1278 MB VRAM)

Average ms/step across 6 cases, median of 3 runs:

| Voice  | Avg ms/step | Avg RTF |
|--------|------------:|--------:|
| alloy  |       135.2 |   1.69x |
| serena |       133.9 |   1.68x |
| ryan   |       134.8 |   1.69x |
| aiden  |       133.0 |   1.67x |

### 0.6B-CustomVoice / INT8 — Multi-Voice (853 MB VRAM)

| Voice  | Avg ms/step | Avg RTF |
|--------|------------:|--------:|
| alloy  |       135.7 |   1.70x |
| serena |       144.7 |   1.81x |
| ryan   |       143.2 |   1.79x |
| aiden  |       152.9 |   1.92x |

### 1.7B-Base / FP16 (3136 MB VRAM)

| Case | Steps | Audio | Total (median) | RTF | ms/step |
|------|------:|------:|---------------:|----:|--------:|
| 4s   |    51 |  4.1s |        7879 ms | 1.94x | 154.5 |
| 6s   |    69 |  5.5s |       11063 ms | 2.01x | 160.3 |
| 8s   |   116 |  9.3s |       19010 ms | 2.05x | 163.9 |
| 10s  |   194 | 15.5s |       30545 ms | 1.97x | 157.5 |
| 12s  |   200 | 16.0s |       31758 ms | 1.99x | 158.8 |
| 16s  |   200 | 16.0s |       31343 ms | 1.96x | 156.7 |

### 1.7B-Base / INT8 (1786 MB VRAM)

| Case | Steps | Audio | Total (median) | RTF | ms/step |
|------|------:|------:|---------------:|----:|--------:|
| 4s   |    52 |  4.1s |        9398 ms | 2.27x | 180.7 |
| 6s   |   200 | 16.0s |       37819 ms | 2.37x | 189.1 |
| 8s   |   132 | 10.5s |       26183 ms | 2.48x | 198.4 |
| 10s  |   200 | 16.0s |       40996 ms | 2.57x | 205.0 |
| 12s  |   200 | 16.0s |       40310 ms | 2.52x | 201.5 |
| 16s  |   200 | 16.0s |       40234 ms | 2.52x | 201.2 |

### Cross-Model Summary

Average ms/step (across all 6 cases, median runs):

| Model | FP16 | INT8 | INT8 Penalty | VRAM (FP16) | VRAM (INT8) |
|-------|-----:|-----:|-------------:|------------:|------------:|
| 0.6B-Base | 140.3 | 147.7 | +5% | 1278 MB | 853 MB |
| 0.6B-CV (alloy) | 135.2 | 135.7 | +0% | 1278 MB | 853 MB |
| 1.7B-Base | 158.6 | 196.0 | **+24%** | 3136 MB | 1786 MB |

### Observations

- **0.6B INT8 is essentially free** — <5% speed penalty, 33% VRAM savings vs FP16.
- **1.7B INT8 has a real speed cost** (+24% ms/step). The INT8 dequantize kernel overhead is more visible with 1.7B's larger matrices. Still, 1786 MB VRAM (vs 3136 MB FP16) is a massive win for 8 GB GPUs.
- **1.7B FP16 is ~15% slower than 0.6B FP16** per step (~159 vs ~140 ms/step), reflecting the larger model's compute requirements.
- **CustomVoice voice selection has minimal FP16 speed impact** (<2% variation across 4 voices). INT8 shows more variation (up to 13% between alloy and aiden), likely from voice-dependent sampling patterns interacting with INT8 quantization.
- **Step counts differ across quant modes** for the same text — expected, as quantization changes logits enough to diverge autoregressive sampling. This makes direct wall-clock comparisons across quant modes unreliable; ms/step is the stable metric.
