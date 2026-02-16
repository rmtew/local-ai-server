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

---

## 2026-02-16 — GPU Encoder (CUDA kernels + cuBLAS Conv2D)

**Changes:**
1. Custom CUDA kernels for encoder LayerNorm, GELU, bias_add
2. Device-to-device cuBLAS GEMMs for transformer layers (attention still on CPU)
3. cuBLAS for Conv2D stem GEMMs (im2col on CPU, GEMM on GPU)

### Results (median of 5 runs, 2 warmup)

| Sample | Audio | Total (ms) | Encode (ms) | Decode (ms) | Wall (ms) | RTF | Words |
|--------|------:|-----------:|------------:|------------:|----------:|----:|------:|
| test_speech.wav | 3.8s | 339 | 78 | 258 | 656 | 0.09x | 15 |
| jfk.wav | 11.0s | 783 | 231 | 552 | 1103 | 0.07x | 26 |

### Comparison vs Baseline

| Sample | Encode Before | Encode After | Delta | Total Before | Total After | Delta |
|--------|------------:|------------:|------:|------------:|------------:|------:|
| test_speech.wav | 153ms | 78ms | **-49%** | 420ms | 339ms | **-19%** |
| jfk.wav | 522ms | 231ms | **-56%** | 1111ms | 783ms | **-30%** |

### Analysis

The biggest win came from moving Conv2D stem GEMMs to cuBLAS. Profiling with verbose=2 showed:
- Transformer layers (CUDA kernels + d2d GEMMs): only 22ms/59ms
- Conv2D stem (CPU OpenBLAS): **127ms/372ms** — the actual bottleneck

Conv2 GEMM per chunk: [480, 4320] @ [4320, 800] = ~1.7 GFLOPS. With 11 chunks for jfk, that's ~18 GFLOPS total — near OpenBLAS peak (~100 GFLOPS) but trivial for cuBLAS (~5 TFLOPS).

RTF improved from 0.10x to 0.07x (14x real-time). Encoder is no longer the dominant bottleneck for short/medium audio.

---

## 2026-02-16 — FP16 Decoder Weights (VRAM savings)

**Change:** When `--fp16` is set, ASR decoder weights (BF16 on disk) are uploaded to GPU as FP16 instead of F32, halving decoder VRAM. Encoder weights (F32 on disk) remain F32.

**Config:** Qwen3-ASR-0.6B, 4 threads, RTX 3070 Laptop GPU, `--fp16`

### VRAM Impact

| Mode | ASR VRAM | Weights | Breakdown |
|------|----------|---------|-----------|
| F32 (default) | 3655 MB | 339 weights, all F32 | — |
| FP16 | 2182 MB | 339 weights: 3 F32, 336 FP16 | **-1473 MB (-40%)** |

The 3 F32 weights are encoder weights (loaded from F32 safetensors). All 336 decoder weights (loaded from BF16 safetensors) are stored as FP16.

### Performance (2 iterations)

| Mode | test_speech.wav Decode | jfk.wav Decode | Transcription |
|------|----------------------:|---------------:|---------------|
| F32 (full GPU decoder) | 267ms (0.09x RTF) | 575ms (0.07x RTF) | Correct |
| FP16 (CPU decoder + GPU GEMM) | 549ms (0.17x RTF) | 1116ms (0.13x RTF) | Correct |

### Analysis

**The full GPU decoder does not work with FP16 weights.** Three approaches were tested:

1. **Mixed FP16×F32 inputs** to `cublasGemmEx`: Returns `CUBLAS_STATUS_SUCCESS` but produces all zeros. cuBLAS does not support mixed input types — both A and B must have the same datatype.

2. **FP16×FP16 with activation conversion** (D2H → F32→F16 → H2D → `cublasGemmEx`): Produces non-zero output but garbage tokens (99889, 119983, etc.). The numeric variance from CUDA kernel non-GEMM ops (RMSNorm, RoPE, SwiGLU) compounds with FP16 quantization loss through 24 autoregressive decoder layers, causing token divergence.

3. **Delegation to `qwen_gpu_gemm`** (known-working CPU↔GPU GEMM path): Same garbage tokens. Confirms the issue is not in the GEMM itself but in the interaction between GPU-side non-GEMM ops and reduced precision.

**Working solution:** When FP16 is active, skip the full GPU decoder and fall back to the CPU decoder with GPU GEMM offload (`qwen_gpu_gemm`). The CPU handles non-GEMM ops (RMSNorm, RoPE, attention, SwiGLU) in F32, while GEMMs are offloaded to GPU using FP16 weights via `cublasGemmEx` (FP16×FP16→F32 accumulation). This preserves the VRAM savings (~1.5 GB) with correct transcription, at ~2x decode slowdown.

**Trade-off:** 40% VRAM reduction (3655→2182 MB) for ~2x slower decode. Total ASR time (including encoder, which is unchanged) goes from ~0.08x to ~0.15x RTF — still 6-7x faster than real-time. Worthwhile when VRAM is scarce (e.g. running both ASR + TTS 1.7B simultaneously).

---

## 2026-02-17 — FP16 On-Device Dequantization (CUDA 13.1)

**Change:** Replace the FP16 CPU decoder fallback with on-device FP16→F32 dequantization. Weights stay as FP16 on GPU (VRAM savings preserved), but are dequantized into a scratch buffer before each GEMM, allowing standard `cublasSgemm` (F32×F32). The full GPU decoder is now used for both F32 and FP16 modes.

**Config:** Qwen3-ASR-0.6B, 4 threads, RTX 3070 Laptop GPU, CUDA 13.1, `--fp16-asr`

### Approach

The previous FP16 approach failed because FP16×FP16 GEMMs compound precision errors through 28 decoder layers when combined with CUDA kernel numeric variance (RMSNorm, RoPE, SwiGLU), producing garbage tokens. The CPU fallback worked but was ~2x slower.

**Solution:** On-device dequantization eliminates this problem:
1. Store weights as FP16 on GPU (same VRAM savings as before)
2. Before each GEMM, launch a CUDA kernel (`qwen_f16_to_f32`) that converts the FP16 weight tile to F32 in a pre-allocated scratch buffer
3. Run standard `cublasSgemm` with F32 weights × F32 activations
4. No activation precision loss, no PCIe round-trips, full GPU decoder stays enabled

### VRAM Impact (0.6B)

| Component | Size |
|-----------|------|
| FP16 weights on device | 2182 MB (unchanged from FP16 mode) |
| Dequant scratch buffer | ~25 MB (gate_up_fused: 2 × intermediate × hidden) |
| **Total** | **~2207 MB** |
| **Savings vs F32** | **~1448 MB (40%)** |

The lm_head (151936 × 1024 = 593 MB F32) is tiled through the 25 MB scratch buffer in ~25 chunks. Per-layer GEMMs dequant in a single shot (all fit within the scratch buffer).

### Performance (median of 5 runs, 2 warmup, CUDA 13.1)

| Mode | test_speech.wav | | jfk.wav | |
|------|-----:|-----:|-----:|-----:|
| | Decode | Total | Decode | Total |
| F32 (full GPU decoder) | 314ms | 398ms | 709ms | 951ms |
| FP16 (dequant + cublasSgemm) | 673ms | 771ms | 1328ms | 1599ms |
| FP16 (fused scalar matvec) | 469ms | 558ms | 1036ms | 1284ms |
| FP16 (fused vectorized matvec) | 418ms | 521ms | 885ms | 1137ms |

Transcription: all modes produce identical, correct output.

### Dequant-only analysis (intermediate step)

The dequant approach was **~2x slower for decode** than F32. Root cause: memory bandwidth.

- **F32 matvec:** read 4 bytes/element from weight → GEMM (bandwidth-limited for M=1)
- **FP16 dequant matvec:** read 2 bytes (FP16) + write 4 bytes (F32 scratch) + read 4 bytes (F32 for GEMM) = 10 bytes/element, **2.5× the bandwidth**

### Fused matvec kernel

Replaced dequant+cublasSgemm with a custom CUDA kernel (`qwen_fp16_matvec_f32`) for M=1 decode:

1. One block per output element (N blocks, 256 threads each)
2. Activation vector loaded into shared memory (cooperative via 128-bit float4 loads, L2-cached across blocks)
3. Each thread reads FP16 weight elements via 128-bit uint4 loads (8 FP16 per load), converts to F32 in registers using branchless bit manipulation, multiplies by cached activation, accumulates
4. Block-level reduction → single output element

Bandwidth: **2 bytes/element** (FP16 weight read only — half of F32's 4 bytes/element).

### Optimization progression

| | test_speech.wav Decode | jfk.wav Decode |
|---|---:|---:|
| Dequant → scalar fused | **-30%** (673→469ms) | **-22%** (1328→1036ms) |
| Scalar → vectorized fused | **-11%** (469→418ms) | **-15%** (1036→885ms) |
| **Total: dequant → vectorized** | **-38%** (673→418ms) | **-33%** (1328→885ms) |
| Vectorized fused vs F32 | **+33%** (314→418ms) | **+25%** (709→885ms) |

The vectorized kernel narrows the gap with F32 cuBLAS to **~30% overhead** (down from ~50% scalar, ~100% dequant). The remaining gap is cuBLAS's highly tuned memory access patterns and instruction scheduling for bandwidth-limited matvecs.

### Warp shuffle reduction

Replaced the final 5 `__syncthreads()` barriers in the block reduction with `__shfl_down_sync()` warp intrinsics. Back-to-back benchmark (same thermal window):

| Mode | test_speech.wav | | jfk.wav | |
|------|-----:|-----:|-----:|-----:|
| | Decode | Total | Decode | Total |
| F32 (full GPU decoder) | 300ms | 386ms | 614ms | 904ms |
| FP16 (vectorized + warp shuffle) | 297ms | 394ms | 630ms | 910ms |

FP16 overhead vs F32: **-1% / +3%** (within noise). The combination of vectorized loads, shared memory caching, and warp shuffle brings FP16 decode to effective parity with F32 cuBLAS.

**Trade-off summary:** 40% VRAM savings (3655→2207 MB) for negligible decode overhead. RTF 0.08-0.10x (10-12× real-time). The dequant+cublasSgemm path remains as fallback for M>1 (not used in the decoder path).

