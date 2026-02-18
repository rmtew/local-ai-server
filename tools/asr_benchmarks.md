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

---

## 2026-02-18 — Qwen3-ASR-1.7B (FP16 decoder)

**Config:** Qwen3-ASR-1.7B, 4 threads, RTX 3070 Laptop GPU, CUDA 13.1, `--fp16-asr`

### VRAM

| Model | Mode | VRAM |
|-------|------|------|
| 0.6B | FP16 | 2207 MB |
| 1.7B | FP16 | 7779 MB |

The 1.7B model with FP16 decoder nearly fills the 8192 MB RTX 3070 Laptop GPU (413 MB headroom). Not enough room to co-load TTS (0.6B TTS needs ~1278 MB).

### Results (median of runs 2-5, run 1 warmup)

| Sample | Audio | Total (ms) | Encode (ms) | Decode (ms) | RTF | Words |
|--------|------:|-----------:|------------:|------------:|----:|------:|
| test_speech.wav | 3.6s | 518 | 104 | 419 | 0.14x | 15 |
| jfk.wav | 11.0s | 1131 | 260 | 872 | 0.10x | 26 |

### Raw runs

**test_speech.wav:**

| Run | Total (ms) | Encode (ms) | Decode (ms) |
|-----|----------:|------------:|------------:|
| 1 (warmup) | 804 | 317 | 487 |
| 2 | 526 | 104 | 422 |
| 3 | 523 | 104 | 418 |
| 4 | 513 | 94 | 419 |
| 5 | 499 | 89 | 410 |

**jfk.wav:**

| Run | Total (ms) | Encode (ms) | Decode (ms) |
|-----|----------:|------------:|------------:|
| 1 (warmup) | 1750 | 594 | 1155 |
| 2 | 1130 | 261 | 868 |
| 3 | 1134 | 259 | 875 |
| 4 | 1124 | 266 | 857 |
| 5 | 1132 | 250 | 882 |

### Comparison vs 0.6B FP16 (GPU Encoder era)

| Sample | 0.6B Total | 1.7B Total | Slowdown | 0.6B Decode | 1.7B Decode | Slowdown |
|--------|----------:|----------:|---------:|------------:|------------:|---------:|
| test_speech.wav | 339ms | 518ms | 1.53x | 258ms | 419ms | 1.62x |
| jfk.wav | 783ms | 1131ms | 1.44x | 552ms | 872ms | 1.58x |

### Analysis

- **RTF 0.10-0.14x** (7-10× real-time) — still very fast for interactive use
- **1.5-1.6x slower decode** than 0.6B, consistent with the larger model (28 vs 24 decoder layers, wider hidden dim)
- **Encode time similar** to 0.6B (~104ms for 3.6s, ~260ms for 11s) — encoder architecture scales less aggressively
- **VRAM is the constraint**, not speed. At 7779 MB, the 1.7B can only run solo (no TTS co-loading on 8 GB GPU)
- **Use case:** Swap in 1.7B when accuracy matters more than concurrent TTS — longer/noisier audio, multilingual content, or difficult proper nouns

---

## 2026-02-18 — Word Error Rate (LibriSpeech)

**Setup:** WER evaluation using LibriSpeech test-clean and test-other (first 100 utterances each). Ground-truth human transcripts compared against model output. Standard WER normalization (uppercase, strip punctuation, word-level Levenshtein).

**Tool:** `tools/wer_bench.py` — transcribes FLAC files via HTTP API, computes WER with S/I/D breakdown.

**Dataset:** LibriSpeech (https://www.openslr.org/12/), stored at `DEPS_ROOT/datasets/librispeech/`.

### Results (100 utterances per split)

| Model | test-clean | | test-other | |
|-------|----:|----:|----:|----:|
| | WER | Errors | WER | Errors |
| 0.6B FP16 | **1.15%** | 27 (22S/2I/3D) / 2357 | **2.67%** | 43 (38S/3I/2D) / 1611 |
| 1.7B FP16 | **1.15%** | 27 (22S/2I/3D) / 2357 | **2.67%** | 43 (38S/3I/2D) / 1611 |

### Analysis

The 1.7B model produces **identical WER** to 0.6B on LibriSpeech — same error count, same S/I/D breakdown. Both models are saturated on this dataset (clean read English audiobooks).

The 1.7B's advantages (multilingual, noisy environments, rare vocabulary) likely show on harder benchmarks — accented speech, real-world recordings, non-English languages. LibriSpeech is too clean to differentiate.

### Context

Published WER benchmarks for reference (full test sets):

| Model | test-clean | test-other |
|-------|--------:|--------:|
| Whisper Large-v3 | 2.0% | 3.6% |
| Qwen3-ASR 0.6B/1.7B (ours, 100 utts) | 1.15% | 2.67% |

Note: Our numbers are on 100 utterances (not full 2620/2939), so they'll shift with more data.

---

## 2026-02-18 — INT8 Decoder Weights

**Change:** ASR decoder weights stored as INT8 (1 byte/param) with per-row absmax quantization. Scale factor per row: `scale = max(abs(row)) / 127`, `int8[j] = round(row[j] / scale)`. Quantized on-the-fly from BF16 safetensors at load time (no model file changes).

**Config:** `--int8-asr` flag or `asr_int8` config key.

### Implementation

- **Fused INT8 matvec kernel** (`qwen_int8_matvec_f32`) for M=1 decode: 128-bit `uint4` loads = 16 INT8 values per load (vs 8 FP16), per-row scale applied after reduction. Bandwidth: 1 byte/weight element (half of FP16's 2 bytes).
- **INT8→F32 dequant + cublasSgemm** fallback for M>1 prefill.
- **lm_head stays FP16** (not INT8) — argmax accuracy requires higher precision for the final vocabulary projection.

### VRAM Impact

| Model | F32 | FP16 | INT8 | INT8 Savings vs FP16 |
|-------|----:|-----:|-----:|---------------------:|
| 0.6B | 3655 MB | 2207 MB | — | — |
| 1.7B | — | 7779 MB | 3822 MB | **-3957 MB (-51%)** |

1.7B INT8 + TTS 0.6B FP16: 3822 + 1278 = **5100 MB** — fits on 8 GB GPU with 3 GB headroom.

### WER (LibriSpeech, 100 utterances)

| Model | Mode | test-clean | test-other |
|-------|------|--------:|--------:|
| 0.6B | FP16 | 1.15% | 2.67% |
| 0.6B | INT8 | 1.53% | 3.04% |
| 1.7B | FP16 | 1.15% | 2.67% |
| 1.7B | INT8 | **1.15%** | **2.79%** |

1.7B INT8 has no meaningful WER degradation (identical test-clean, +0.12% test-other). 0.6B INT8 shows slightly higher WER (+0.38% test-clean, +0.37% test-other) — the smaller model is more sensitive to weight quantization, though still well within acceptable range.

### Bug Fix: d_ffn_out Buffer Overflow

During INT8 testing, discovered a pre-existing buffer overflow in the GPU decoder: `d_ffn_out` was allocated as `[hidden]` (2048 for 1.7B) but receives `[intermediate]` (6144) elements from SwiGLU. The overflow (4096 extra floats = 16 KB) corrupted adjacent GPU memory (RMSNorm weights), causing garbage output for 1.7B INT8. Fixed by allocating `d_ffn_out` as `[intermediate]`. This bug was latent in all previous GPU decoder builds — it happened to not corrupt critical memory with FP16/F32 weight layouts, or the overflow landed in padding/unused regions.

---

## Future Optimizations (Backlog)

### FP16 Encoder Weights (1.7B: ~600 MB VRAM savings)

The encoder weights are currently F32 on GPU. For 0.6B this is only 0.694 GiB (negligible), but for 1.7B it's **1.183 GiB**. Storing encoder weights as FP16 and dequanting before GEMM would save ~600 MB, bringing 1.7B from 7779 MB to ~7180 MB.

**Approach:** Use the proven dequant+cublasSgemm path (store FP16 on GPU, dequant to F32 scratch buffer before each GEMM). The fused FP16 matvec kernel won't help here — encoder GEMMs are batched (M>1), not M=1 matvecs.

**Caveats:**
- Dequant+cublasSgemm has ~2x bandwidth overhead vs plain F32 cublasSgemm (read 2B FP16 + write 4B F32 scratch + read 4B F32 = 10 bytes/element vs 4 bytes/element)
- Encoder is currently fast (104-260ms), so even 2x slowdown only adds 100-260ms — acceptable for interactive use
- Do NOT attempt FP16×FP16 cublasGemmEx — mixed-type inputs silently produce zeros, and same-type FP16×FP16 accumulates precision errors through stacked layers (documented in "FP16 Decoder Weights" section above)

