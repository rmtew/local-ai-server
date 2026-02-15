# Qwen3-TTS Integration

This document describes how Qwen3-TTS text-to-speech is implemented in local-ai-server, what models are required, and how the pipeline works.

## Overview

Qwen3-TTS converts text to speech in three stages:

1. **Talker LM** -- autoregressive transformer that generates codec tokens from text
2. **Code Predictor** -- predicts sub-codebook tokens for each step
3. **Vocoder** -- converts codec tokens to 24 kHz audio waveform

All three stages run natively in C. The talker and code predictor use cuBLAS GPU acceleration. The vocoder uses OpenBLAS CPU acceleration.

## Required Models

Two model directories are needed, expected as siblings on disk:

```
models/tts/
  qwen3-tts-12hz-0.6b-base/     <-- passed via --tts-model
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

- `model.safetensors` (1.9 GB) from [`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) -- talker + code predictor + speaker encoder weights
- `model.safetensors` (682 MB) from [`Qwen/Qwen3-TTS-Tokenizer-12Hz`](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz) -- vocoder weights
- Tokenizer files (vocab.json, merges.txt, etc.) from [`zukky/Qwen3-TTS-ONNX-DLL`](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL)

## Pipeline Architecture

### Stage 1: Text Tokenization and Embedding

Input text is tokenized using the Qwen2 BPE tokenizer (151,643 tokens). Token embeddings are computed via a two-layer text projection (native C) and combined with codec prefix tokens (`nothink`, `think_bos`, `think_eos`, `pad`, `bos`).

### Stage 2: Talker LM (Native C + cuBLAS)

A 28-layer Qwen3 transformer generates codec tokens autoregressively:
- Hidden size: 1024, 16 attention heads, GQA with 4 KV heads
- Sampling: top-k (k=50) with temperature 0.9 (configurable) and repetition penalty 1.05
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

| Stage | Time |
|-------|------|
| Talker decode (GPU) | ~1.6s |
| Native vocoder (CPU) | ~10.1s |
| **Total** | ~11.7s |

### Medium sentence (~2 seconds of audio, ~24 codec steps)

| Stage | Time |
|-------|------|
| Talker decode (GPU) | ~3.5s |
| Native vocoder (CPU) | ~25s |
| **Total** | ~29s |

The vocoder is the bottleneck -- BigVGAN Conv1d operations account for ~96% of vocoder time. Further optimization with AVX2/SSE vectorization and BLAS-accelerated convolutions is planned.

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `tts_pipeline.c` | ~290 | Pipeline orchestration, WAV encoding |
| `tts_native.c` | ~1900 | Native talker LM + code predictor (C + cuBLAS) |
| `tts_vocoder.c` | ~870 | Vocoder pipeline, weight loading, RVQ decode, buffer management |
| `tts_vocoder_ops.c` | ~230 | Conv1d, ConvTranspose1d, SnakeBeta, LayerNorm, GELU |
| `tts_vocoder_xfmr.c` | ~240 | 8-layer pre-transformer |
| `tts_vocoder.h` | ~200 | Vocoder types, constants, weight structs |
| `tts_sampling.c` | ~115 | Top-k sampling, repetition penalty |

## Voice Cloning

Our model IS the Base variant (`Qwen3-TTS-12Hz-0.6B-Base`), which includes 76 `speaker_encoder.*` tensors (~17 MB ECAPA-TDNN) alongside the talker and code predictor weights. Voice cloning uses precomputed speaker embeddings stored in `voice_presets.bin`.

### How It Works

The [qwen-tts Python package](https://github.com/QwenLM/Qwen3-TTS) supports two voice cloning modes:

1. **X-vector only** (`x_vector_only_mode=True`): Reference audio is processed through the speaker encoder to produce a 1024-dim embedding. This embedding is inserted as a single extra token in the talker LM's input sequence. No reference text needed. **This is the mode we implement.**

2. **Full in-context learning** (`x_vector_only_mode=False`): Reference audio is tokenized through the codec, reference text is encoded, and both are prepended to the input alongside the speaker embedding. Higher quality, requires reference transcript. Not yet implemented.

### Speaker Encoder Architecture (ECAPA-TDNN)

Implemented in `tts_speaker_enc.c/.h`. Weights loaded from `speaker_encoder.*` tensors in `model.safetensors`.

```
Audio (24 kHz)
  |
  v
Mel spectrogram (tts_mel.c: n_fft=1024, hop=256, win=1024, 128 mels, fmin=0, fmax=12000)
  |
  v
Block 0: Conv1d(128->512, k=5) + ReLU
  |
  v
Blocks 1-3: SE-Res2Net (scale=8, dilations 2/3/4, squeeze-excitation channel attention)
  |
  v
MFA: concat(block1,block2,block3) -> Conv1d(1536->1536, k=1) + ReLU
  |
  v
ASP: Attentive Statistics Pooling (attention-weighted mean + std)
  |
  v
FC: Conv1d(3072->1024, k=1) -> 1024-dim speaker embedding
```

### Speaker Embedding Injection

The 1024-dim embedding is inserted as a single token between the two parts of the codec prefix in `build_prefill_embeddings()`:

```
Without speaker:  [think, think_bos, lang_id, think_eos, pad, bos]
With speaker:     [think, think_bos, lang_id, think_eos, SPEAKER, pad, bos]
```

Text side pairs the speaker position with a `tts_pad` embedding. Prefill length increases by 1.

### Voice Presets

Precomputed 1024-dim embeddings stored in `<tts-model-dir>/voice_presets.bin`. Loaded at startup by `tts_voice_presets.c`. The `voice` parameter in the API selects a preset by name.

Generate presets from reference WAV files:
```bash
python tools/generate_voice_presets.py --samples-dir voice_samples/ --output voice_presets.bin
```
