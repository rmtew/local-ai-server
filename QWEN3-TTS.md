# Qwen3-TTS Integration

This document describes how Qwen3-TTS text-to-speech is implemented in local-ai-server, what models are required, and how the pipeline works.

## Overview

Qwen3-TTS converts text to speech in three stages:

1. **Talker LM** -- autoregressive transformer that generates codec tokens from text
2. **Code Predictor** -- predicts sub-codebook tokens for each step
3. **Vocoder** -- converts codec tokens to 24 kHz audio waveform

All three stages run natively in C. The talker and code predictor use cuBLAS GPU acceleration. The vocoder uses OpenBLAS CPU acceleration. ONNX Runtime is linked but only used for optional future features (voice cloning).

## Required Models

Two model directories are needed, expected as siblings on disk:

```
models/tts/
  qwen3-tts-0.6b/               <-- passed via --tts-model
    model.safetensors            (1.9 GB) -- talker + code predictor weights
    config.json
    vocab.json
    merges.txt
    tokenizer_config.json
  Qwen3-TTS-Tokenizer-12Hz/     <-- auto-discovered as sibling
    model.safetensors            (682 MB) -- vocoder weights
```

The server expects both directories. The vocoder weights are discovered at `<tts-model-dir>/../Qwen3-TTS-Tokenizer-12Hz/model.safetensors`.

### Downloading Models

All models can be downloaded with the included script:

```bash
# Requires DEPS_ROOT environment variable
bash tools/download_tts_models.sh
```

This downloads everything needed for native inference:

- `model.safetensors` (1.9 GB) from [`Qwen/Qwen3-TTS-0.6B`](https://huggingface.co/Qwen/Qwen3-TTS-0.6B) -- talker + code predictor weights
- `model.safetensors` (682 MB) from [`Qwen/Qwen3-TTS-Tokenizer-12Hz`](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz) -- vocoder weights
- Tokenizer files (vocab.json, merges.txt, etc.) from [`zukky/Qwen3-TTS-ONNX-DLL`](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL)

## Pipeline Architecture

### Stage 1: Text Tokenization and Embedding

Input text is tokenized using the Qwen2 BPE tokenizer (151,643 tokens). Token embeddings are computed via a two-layer text projection (native C) and combined with codec prefix tokens (`nothink`, `think_bos`, `think_eos`, `pad`, `bos`).

### Stage 2: Talker LM (Native C + cuBLAS)

A 28-layer Qwen3 transformer generates codec tokens autoregressively:
- Hidden size: 1024, 16 attention heads, GQA with 4 KV heads
- Sampling: top-k (k=50) with temperature 0.3 and repetition penalty 1.05
- Each step produces one first-codebook token (from vocabulary of 2048)

Implementation: `src/tts_native.c` (~1900 lines). Weights loaded from `model.safetensors` (BF16, converted to FP32 at load). KV cache maintained across steps.

### Stage 3: Code Predictor (Native C + cuBLAS)

For each talker step, a 5-layer transformer predicts 15 additional sub-codebook tokens:
- Hidden size: 1024, 16 heads
- One forward pass per sub-code, with the growing sequence of prior sub-codes as input

Total output: `[n_steps, 16]` int64 codec tokens.

### Stage 4: Vocoder (Native C + OpenBLAS)

Converts codec tokens to a 24 kHz audio waveform. This is the most compute-intensive stage.

Implementation: `src/tts_vocoder.c`, `src/tts_vocoder_ops.c`, `src/tts_vocoder_xfmr.c` (~1750 lines total). Weights loaded from `Qwen3-TTS-Tokenizer-12Hz/model.safetensors` (FP32).

#### Vocoder Sub-stages

```
codes [T, 16]
  |
  v
RVQ Decode: 16 codebook lookups (2048x256) + 1x1 Conv projections -> [512, T]
  |
  v
Pre-Conv: CausalConv1d(512, 1024, k=3) -> [1024, T]
  |
  v
Pre-Transformer: 8-layer Qwen2 (hidden=512, 16 heads, head_dim=64, RoPE theta=10k)
  input_proj(1024->512) -> 8x [RMSNorm, Attention, LayerScale, RMSNorm, SwiGLU, LayerScale]
  -> final_norm -> output_proj(512->1024) -> [1024, T]
  |
  v
ConvNeXt Upsample: 2 stages (2x each) -> [1024, 4T]
  Each: CausalConvTranspose1d(k=2, s=2) + ConvNeXt(DWConv k=7, LayerNorm, GELU MLP, gamma)
  |
  v
BigVGAN Decoder: 4 blocks -> [1, 1920T]
  Init: CausalConv1d(1024, 1536, k=7)
  Block rates: [8, 5, 4, 3], channels: 1536 -> 768 -> 384 -> 192 -> 96
  Each: SnakeBeta -> CausalConvTranspose1d(k=2r, s=r) -> 3x ResUnit(dilations 1,3,9)
  Final: SnakeBeta(96) -> CausalConv1d(96, 1, k=7) -> clamp(-1, 1)
```

Total upsample factor: 2 * 2 * 8 * 5 * 4 * 3 = 1920x (12.5 Hz codec rate to 24 kHz audio).

#### Key Operations

- **CausalConv1d**: Left-padded by `dilation * (kernel-1)`, then im2col + SGEMM (OpenBLAS). Depthwise variant uses direct loop.
- **CausalConvTranspose1d**: Scatter-add with causal trim of `kernel-stride` from each side. Output length = `(T+1)*stride - kernel`.
- **SnakeBeta**: `x += (1/exp(beta)) * sin^2(exp(alpha) * x)` -- exp values precomputed at load time.
- **LayerNorm**: Per-timestep normalization across channels (for ConvNeXt).
- **GELU**: Exact form using `erff`.

## Performance

Benchmarks on an RTX 3080 + i7-10700K system. Times vary with input length.

### Short sentence (~1 second of audio, ~11 codec steps)

| Configuration | Decode | Vocoder | Total |
|---------------|--------|---------|-------|
| Full ONNX (CPU) | ~1.6s | ~36.7s | ~38.3s |
| Native decode + ONNX vocoder | ~1.6s | ~36.7s | ~38.3s |
| Native decode + native vocoder | ~1.6s | ~10.1s | ~11.7s |

### Medium sentence (~2 seconds of audio, ~24 codec steps)

| Stage | Time |
|-------|------|
| Talker decode (GPU) | ~3.5s |
| Native vocoder (CPU) | ~25s |
| **Total** | ~29s |

The native C vocoder is **3.6x faster** than ONNX Runtime for the vocoder stage. It matches ONNX output exactly (correlation 1.0, SNR > 110 dB).

The vocoder remains the bottleneck -- BigVGAN Conv1d operations account for ~96% of vocoder time. Further optimization with AVX2/SSE vectorization and BLAS-accelerated convolutions is planned.

Note: ONNX Runtime with CUDA ExecutionProvider was tested but was ~2x slower than CPU due to excessive host-device memory copies in the vocoder's many small convolutions.

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `tts_pipeline.c` | ~290 | Pipeline orchestration, WAV encoding |
| `tts_native.c` | ~1900 | Native talker LM + code predictor (C + cuBLAS) |
| `tts_vocoder.c` | ~870 | Vocoder pipeline, weight loading, RVQ decode, buffer management |
| `tts_vocoder_ops.c` | ~230 | Conv1d, ConvTranspose1d, SnakeBeta, LayerNorm, GELU |
| `tts_vocoder_xfmr.c` | ~240 | 8-layer pre-transformer |
| `tts_vocoder.h` | ~200 | Vocoder types, constants, weight structs |
| `tts_ort.c` | ~115 | ONNX Runtime initialization (for future voice cloning) |
| `tts_sampling.c` | ~115 | Top-k sampling, repetition penalty |
