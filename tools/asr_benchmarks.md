# ASR Benchmarks

Performance tracking for ASR inference on local-ai-server.

## Hardware

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8192 MB VRAM, compute 8.6)
- CPU: 4 threads used for inference
- OS: Windows 11
- Build: MSVC /O2 /arch:AVX2 /fp:fast, cuBLAS + custom CUDA decoder kernels

## Test Files

| File | Audio Duration | Content |
|------|---------------|---------|
| test_speech.wav | 3.6s | "Hello. This is a test of the Vox Troll speech-to-text system." |
| jfk.wav | 11.0s | JFK "ask not what your country can do for you" excerpt |

---

## 2026-02-16 — Baseline (0.6B, cuBLAS + CUDA kernels)

**Config:** Qwen3-ASR-0.6B, 4 threads, RTX 3070 Laptop GPU

### Summary (median of 5 runs, 2 warmup)

| Sample | Audio | Total (ms) | Encode (ms) | Decode (ms) | Wall (ms) | RTF | Words |
|--------|------:|-----------:|------------:|------------:|----------:|----:|------:|
| test_speech.wav | 3.6s | 420 | 153 | 264 | 744 | 0.11x | 15 |
| jfk.wav | 11.0s | 1111 | 522 | 588 | 1450 | 0.10x | 26 |

### Detailed Stage Breakdown (verbose=2, warmed-up single run)

**test_speech.wav (3.6s audio -> 47 encoder tokens, 19 decode tokens):**

| Stage | Time | % |
|-------|-----:|--:|
| Mel spectrogram (364 frames) | 5ms | 1% |
| Encoder (47 tokens) | 155ms | 36% |
| GPU KV cache sync (13.3 MB) | ~2ms | <1% |
| Prefill (62 tokens) | 100ms | 23% |
| Decode (19 tokens @ 9.0 ms/tok) | 171ms | 39% |
| **Total** | **~435ms** | |

**jfk.wav (11.0s audio -> 143 encoder tokens, 30 decode tokens):**

| Stage | Time | % |
|-------|-----:|--:|
| Mel spectrogram (1100 frames) | 14ms | 1% |
| Encoder (143 tokens) | 524ms | 47% |
| GPU KV cache sync (34.3 MB) | ~5ms | <1% |
| Prefill (158 tokens) | 275ms | 25% |
| Decode (30 tokens @ 11.3 ms/tok) | 340ms | 31% |
| **Total** | **~1108ms** | |

### Observations

- **RTF ~0.10x** = 10x faster than real-time
- **Encoder dominates** for longer audio (36% at 3.6s, 47% at 11s) — runs on GPU (cuBLAS GEMM) but non-GEMM ops (LayerNorm, GELU, windowed attention) are CPU-side with data round-trips
- **Decode rate: 9-11 ms/token** with full GPU decoder (CUDA kernels, KV on device)
- **Decode slows with context length** — 26% slower per-token at 11s vs 3.6s due to growing attention span
- **Mel is negligible** (<2%) despite naive O(N^2) DFT
- **Prefill: 1.6-1.7 ms/token** — efficient batch cuBLAS GEMM

---

## 2026-02-16 — Threaded Encoder Attention

**Change:** Parallelize `qwen_bidirectional_attention` across heads (same pattern as decoder's `qwen_causal_attention`).

### Results (median of 5 runs, 2 warmup)

| Sample | Audio | Total (ms) | Encode (ms) | Decode (ms) | Wall (ms) | RTF | Words |
|--------|------:|-----------:|------------:|------------:|----------:|----:|------:|
| test_speech.wav | 3.6s | 427 | 152 | 269 | 759 | 0.11x | 15 |
| jfk.wav | 11.0s | 1109 | 487 | 624 | 1491 | 0.10x | 26 |

### Comparison vs Baseline

| Sample | Encode Before | Encode After | Delta |
|--------|------------:|------------:|------:|
| test_speech.wav | 153ms | 152ms | -1ms (noise) |
| jfk.wav | 522ms | 487ms | -35ms (7%) |

### Analysis

Negligible improvement. Encoder windowed attention (14 heads, window=104, head_dim=64) is only ~1-2ms of ~500ms encoder time. The actual encoder bottleneck is **108 cuBLAS GEMM round-trips** per forward pass (6 GEMMs/layer x 18 layers), each requiring CPU->GPU activation transfer, GPU GEMM, and GPU->CPU result transfer.

The threading is correct and harmless — helps on CPU-only builds and will matter more for larger models with more heads. But for GPU-accelerated 0.6B, the next impactful optimization is **CUDA kernels for the encoder** to eliminate the GEMM round-trip overhead.
