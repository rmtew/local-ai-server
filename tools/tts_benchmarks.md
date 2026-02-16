# TTS Benchmarks

Performance tracking for TTS inference on local-ai-server.

## Hardware

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8192 MB VRAM, compute 8.6)
- CPU: 4 threads used for vocoder
- OS: Windows 11
- Build: MSVC /O2 /arch:AVX2 /fp:fast, cuBLAS + custom CUDA kernels
- CUDA: 12.8 (cuBLAS 12)

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
