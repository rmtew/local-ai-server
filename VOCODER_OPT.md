# Vocoder Optimization Log

Tracks per-stage timing measurements across optimization iterations.
All measurements use `vocoder-bench.exe` with the same reference codec tokens.

## Test Configuration

- CPU: AMD Ryzen 7 5800H (8C/16T, 3.2 GHz base)
- Reference: bench_ref.codes (50 steps, ~4.0s audio, 95445 samples)
- Runs per measurement: 3 (median reported)

## Baseline (pre-optimization)

Vocoder compiled with `/Od` (debug, no optimization).

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| RVQ decode | 319 | 0.6% |
| Pre-conv | 1 | 0.0% |
| Pre-transformer | 26 | 0.0% |
| Upsample 0 | 361 | 0.7% |
| Upsample 1 | 715 | 1.4% |
| BigVGAN init | 15 | 0.0% |
| BigVGAN block 0 | 13290 | 25.2% |
| BigVGAN block 1 | 14701 | 27.9% |
| BigVGAN block 2 | 13496 | 25.6% |
| BigVGAN block 3 | 10432 | 19.8% |
| Final | 248 | 0.5% |
| **Total** | **52639** | **100%** |

BigVGAN blocks: 51919 ms combined (98.6% of total).
Comparison vs reference: correlation=1.000000, SNR=inf dB, max_diff=0.000000

## Opt 1: Preallocate scratch buffers

Eliminated ~20 malloc/free calls per vocoder run. Peak scratch computed once
in `ensure_buffers()` by simulating T progression through all stages.

| Stage | Before (ms) | After (ms) | Delta |
|-------|-------------|------------|-------|
| RVQ decode | 319 | 318 | -1 |
| Pre-conv | 1 | 1 | 0 |
| Pre-transformer | 26 | 27 | +1 |
| Upsample 0 | 361 | 362 | +1 |
| Upsample 1 | 715 | 721 | +6 |
| BigVGAN init | 15 | 14 | -1 |
| BigVGAN block 0 | 13290 | 13165 | -125 |
| BigVGAN block 1 | 14701 | 13039 | -1662 |
| BigVGAN block 2 | 13496 | 13099 | -397 |
| BigVGAN block 3 | 10432 | 10283 | -149 |
| Final | 248 | 167 | -81 |
| **Total** | **52639** | **50877** | **-1762 (3.3%)** |

Correlation: 1.000000, SNR: inf dB, Max diff: 0.000000

## Opt 2: Compile vocoder with /O2 /arch:AVX2 /fp:fast

Split vocoder sources (tts_vocoder.c, tts_vocoder_ops.c, tts_vocoder_xfmr.c) into
separate optimized compilation group, matching qwen-asr pattern. Enables compiler
auto-vectorization, loop optimization, and fast-math for all vocoder code.

| Stage | Before (ms) | After (ms) | Speedup |
|-------|-------------|------------|---------|
| RVQ decode | 318 | 8 | 40x |
| Pre-conv | 1 | 1 | -- |
| Pre-transformer | 27 | 26 | -- |
| Upsample 0 | 362 | 148 | 2.4x |
| Upsample 1 | 721 | 278 | 2.6x |
| BigVGAN init | 14 | 11 | -- |
| BigVGAN block 0 | 13165 | 2043 | 6.4x |
| BigVGAN block 1 | 13039 | 3122 | 4.2x |
| BigVGAN block 2 | 13099 | 3200 | 4.1x |
| BigVGAN block 3 | 10283 | 3175 | 3.2x |
| Final | 167 | 73 | 2.3x |
| **Total** | **50877** | **11851** | **4.3x** |

Correlation: 1.000000, SNR: 118.7 dB, Max diff: 0.000001

Note: SNR dropped from inf to 118.7 dB due to `/fp:fast` reordering floating-point
operations. Max diff of 1e-6 is inaudible and expected with fast-math.

**Cumulative: 52639 -> 11851 ms (4.4x speedup vs original baseline)**

## Opt 3: AVX2 vectorized SnakeBeta

Replaced scalar `sinf()` in SnakeBeta with AVX2 polynomial sin approximation
(Cephes-style: Cody-Waite range reduction + degree-7 minimax polynomial, 8 floats
per iteration). ~137M sin() calls across the pipeline.

| Stage | Before (ms) | After (ms) | Delta |
|-------|-------------|------------|-------|
| RVQ decode | 8 | 7 | -1 |
| Pre-conv | 1 | 1 | 0 |
| Pre-transformer | 26 | 26 | 0 |
| Upsample 0 | 148 | 119 | -29 |
| Upsample 1 | 278 | 231 | -47 |
| BigVGAN init | 11 | 13 | +2 |
| BigVGAN block 0 | 2043 | 1681 | -362 |
| BigVGAN block 1 | 3122 | 2576 | -546 |
| BigVGAN block 2 | 3200 | 2751 | -449 |
| BigVGAN block 3 | 3175 | 2593 | -582 |
| Final | 73 | 67 | -6 |
| **Total** | **11851** | **10015** | **-1836 (15.5%)** |

Correlation: 1.000000, SNR: 92.4 dB, Max diff: 0.000033

Note: SNR 92.4 dB from polynomial sin approximation (max error ~2e-7 per sin call,
accumulates over ~137M calls). Entirely inaudible -- 92 dB is below 16-bit PCM
quantization noise.

**Cumulative: 52639 -> 10015 ms (5.3x speedup vs original baseline)**

## Sub-block profiling (pre-Opt 4)

Inserted timing between ConvTranspose and ResUnits within each BigVGAN block.

| Block | Total (ms) | tconv (ms) | resunits (ms) | tconv % |
|-------|-----------|-----------|--------------|---------|
| 0 | 1619 | 1518 | 102 | 93.8% |
| 1 | 2608 | 2428 | 203 | 93.1% |
| 2 | 3219 | 2731 | 488 | 84.8% |
| 3 | 2584 | 2130 | 452 | 82.4% |
| **Sum** | **10030** | **8807** | **1245** | **87.8%** |

ConvTranspose1d (naive scalar scatter-add loop) was 88% of BigVGAN time.

## Opt 4: GEMM-based ConvTranspose1d

Replaced naive O(c_in * c_out * T * kernel) scatter-add loop with:
1. GEMM: weight^T [c_out*kernel, c_in] @ input [c_in, T] -> cols [c_out*kernel, T]
2. col2im: scatter-add columns into output (addition only, no multiply)

The GEMM leverages OpenBLAS vectorized SGEMM. The col2im pass is O(c_out*kernel*T)
additions -- negligible vs the O(c_in*c_out*kernel*T) GEMM.

| Stage | Before (ms) | After (ms) | Speedup |
|-------|-------------|------------|---------|
| RVQ decode | 7 | 7 | -- |
| Pre-conv | 1 | 1 | -- |
| Pre-transformer | 23 | 26 | -- |
| Upsample 0 | 117 | 12 | 9.8x |
| Upsample 1 | 218 | 21 | 10.4x |
| BigVGAN init | 9 | 13 | -- |
| BigVGAN block 0 | 1619 | 151 | 10.7x |
|   tconv | 1518 | 30 | 50.6x |
|   resunits | 102 | 119 | -- |
| BigVGAN block 1 | 2608 | 236 | 11.0x |
|   tconv | 2428 | 31 | 78.3x |
|   resunits | 203 | 206 | -- |
| BigVGAN block 2 | 3219 | 545 | 5.9x |
|   tconv | 2731 | 44 | 62.1x |
|   resunits | 488 | 501 | -- |
| BigVGAN block 3 | 2584 | 508 | 5.1x |
|   tconv | 2130 | 51 | 41.8x |
|   resunits | 452 | 457 | -- |
| Final | 63 | 74 | -- |
| **Total** | **10420** | **1589** | **6.6x** |

Correlation: 1.000000, SNR: 92.4 dB, Max diff: 0.000033

ConvTranspose1d went from 8807 ms (88% of total) to 156 ms (10%). Profile has
completely shifted: ResUnits now dominate at 1283 ms (81% of total).

Upsample stages also benefited (335 ms -> 33 ms) since they use the same function.

**Cumulative: 52639 -> 1589 ms (33.1x speedup vs original baseline)**

4.0s audio in 1.6s = **2.5x faster than realtime**.

## Opt 5: Implicit GEMM for large conv1d (eliminate im2col)

For standard (non-depthwise) conv1d with kernel>1, the im2col buffer is
c_in * kernel * T floats. For later BigVGAN blocks this reaches 170-256 MB,
thrashing the entire cache hierarchy.

Replaced with implicit GEMM: `kernel` separate SGEMM calls, each reading
directly from the padded input at offset `ki * dilation`, with row stride
`T_padded`. Eliminates the im2col buffer entirely.

Hybrid threshold at 10M floats (~40MB): below threshold keeps im2col + single
SGEMM (more efficient for small T / large channels where the im2col fits in
L3 cache).

| Stage | Before (ms) | After (ms) | Delta |
|-------|-------------|------------|-------|
| RVQ decode | 7 | 7 | 0 |
| Pre-conv | 1 | 1 | 0 |
| Pre-transformer | 26 | 26 | 0 |
| Upsample 0 | 12 | 13 | +1 |
| Upsample 1 | 21 | 21 | 0 |
| BigVGAN init | 13 | 12 | -1 |
| BigVGAN block 0 | 151 | 159 | +8 |
| BigVGAN block 1 | 236 | 236 | 0 |
| BigVGAN block 2 | 545 | 411 | -134 |
| BigVGAN block 3 | 508 | 416 | -92 |
| Final | 74 | 28 | -46 |
| **Total** | **1589** | **1350** | **-239 (15%)** |

Correlation: 1.000000, SNR: 92.4 dB, Max diff: 0.000033

Blocks 2-3 and Final benefited most (large T, im2col was 170-256 MB).
Block 0 uses im2col path (8.6M < threshold), avoiding a regression.
Peak scratch allocation reduced from ~250 MB to ~75 MB.

**Cumulative: 52639 -> 1350 ms (39.0x speedup vs original baseline)**

4.0s audio in 1.35s = **3.0x faster than realtime**.

## Workflow

```bash
# 1. Generate reference (one-time, --model points to TTS model dir)
bin/vocoder-bench.exe --model=$DEPS_ROOT/models/tts/qwen3-tts-12hz-0.6b-base --generate

# 2. Baseline measurement (--model auto-resolves Tokenizer-12Hz subdir)
bin/vocoder-bench.exe --model=$DEPS_ROOT/models/tts/qwen3-tts-12hz-0.6b-base \
    --codes=bench_ref.codes --ref=bench_ref.raw --runs=3

# 3. After each optimization: rebuild bench, re-run step 2, record results below
```
