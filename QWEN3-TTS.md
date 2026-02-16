# Qwen3-TTS Integration

This document describes how Qwen3-TTS text-to-speech is implemented in local-ai-server, what models are required, and how the pipeline works.

## Overview

Qwen3-TTS converts text to speech in three stages:

1. **Talker LM** -- autoregressive transformer that generates codec tokens from text
2. **Code Predictor** -- predicts sub-codebook tokens for each step
3. **Vocoder** -- converts codec tokens to 24 kHz audio waveform

All three stages run natively in C. The talker and code predictor use cuBLAS GPU acceleration. The vocoder uses OpenBLAS CPU acceleration.

## Required Models

Both 0.6B and 1.7B model sizes are supported. Model size is auto-detected from weight shapes at startup. The vocoder is shared between both models.

```
models/tts/
  qwen3-tts-12hz-0.6b-base/     <-- 0.6B model (pass via --tts-model)
    model.safetensors            (1.9 GB)
    config.json, vocab.json, merges.txt, tokenizer_config.json
  qwen3-tts-12hz-1.7b-base/     <-- 1.7B model (pass via --tts-model)
    model.safetensors            (3.6 GB)
    config.json, vocab.json, merges.txt, tokenizer_config.json
  Qwen3-TTS-Tokenizer-12Hz/     <-- shared vocoder (auto-discovered as sibling)
    model.safetensors            (682 MB)
```

### Architecture Differences

| Parameter | 0.6B | 1.7B |
|-----------|------|------|
| Talker hidden size | 1024 | 2048 |
| Talker intermediate (FFN) | 3072 | 6144 |
| Talker layers / heads / KV heads | 28 / 16 / 8 | 28 / 16 / 8 |
| Code predictor (all dims) | 1024 / 5 layers | 1024 / 5 layers |
| Speaker encoder embed dim | 1024 | 2048 |
| Text hidden size | 2048 | 2048 |
| Head dim / RoPE theta | 128 / 1M | 128 / 1M |

The code predictor is identical between models. The 1.7B model uses `small_to_mtp_projection` (Linear(2048, 1024)) to bridge talker outputs to code predictor inputs. For 0.6B, this projection is identity (same hidden size).

### Downloading Models

Models from HuggingFace:

- **0.6B**: [`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) -- `model.safetensors` (1.9 GB)
- **1.7B**: [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) -- `model.safetensors` (3.6 GB)
- **Vocoder**: [`Qwen/Qwen3-TTS-Tokenizer-12Hz`](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz) -- `model.safetensors` (682 MB)
- Tokenizer files (vocab.json, merges.txt) from [`zukky/Qwen3-TTS-ONNX-DLL`](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL)

## Pipeline Architecture

### Stage 1: Text Tokenization and Embedding

Input text is tokenized using the Qwen2 BPE tokenizer (151,643 tokens). Token embeddings are computed via a two-layer text projection (native C) and combined with codec prefix tokens (`nothink`, `think_bos`, `think_eos`, `pad`, `bos`).

### Stage 2: Talker LM (Native C + cuBLAS)

A 28-layer Qwen3 transformer generates codec tokens autoregressively:
- Hidden size: 1024 (0.6B) or 2048 (1.7B), 16 attention heads, GQA with 8 KV heads, head_dim=128
- Sampling: top-k (k=50) with temperature 0.9 (configurable) and repetition penalty 1.05
- Each step produces one first-codebook token (from vocabulary of 2048)

Implementation: `src/tts_native.c` (~2100 lines). Weights loaded from `model.safetensors` (BF16, converted to F32 at load, or FP16 with `--fp16`). Model size auto-detected from weight shapes. KV cache maintained across steps.

### Stage 3: Code Predictor (Native C + cuBLAS)

For each talker step, a 5-layer transformer predicts 15 additional sub-codebook tokens:
- Hidden size: 1024 (always, for both 0.6B and 1.7B), 16 heads
- For 1.7B: `small_to_mtp_projection` bridges talker (2048-dim) to code predictor (1024-dim)
- Code predictor codec embeddings are in talker space for codec_sum feedback
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

### Streaming (SSE)

When `"stream": true` is set in the request JSON body, the server sends an SSE (Server-Sent Events) response with progress updates during talker decode, followed by base64-encoded WAV audio on completion.

**Event format:**
```
data: {"phase":"decoding","step":1,"max_steps":200}
data: {"phase":"decoding","step":2,"max_steps":200}
...
data: {"phase":"vocoder"}
data: {"phase":"done","n_steps":50,"n_samples":96000,"elapsed_ms":3500.0,"audio":"UklGRi4A..."}
data: [DONE]
```

- `decoding` events fire once per autoregressive step (~100-300ms each)
- `vocoder` fires once when decode completes and vocoder starts
- `done` contains the complete WAV file as a base64-encoded string, plus synthesis metadata
- `[DONE]` is the terminal sentinel (matches OpenAI SSE convention)

**Example (curl):**
```bash
curl -N -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","voice":"alloy","stream":true}'
```

The non-streaming path (`"stream": false` or omitted) is unchanged — returns `audio/wav` binary response directly.

## Performance

All benchmarks on RTX 3070 Laptop GPU (8192 MB VRAM, compute 8.6) with cuBLAS + custom CUDA kernels. Inference is single-threaded with `seed` for deterministic output. Median of 3 runs after warmup.

### GPU VRAM Usage

| Model | Mode | Weights | VRAM | Model load time |
|-------|------|---------|------|-----------------|
| 0.6B | F32 | 216 F32 | **2136 MB** | 3.7s |
| 0.6B | `--fp16` | 171 FP16 + 45 F32 | **1278 MB** | 3.9s |
| 1.7B | F32 | 216 F32 | **5852 MB** | 13.8s |
| 1.7B | `--fp16` | 171 FP16 + 45 F32 | **3136 MB** | 19.4s |

With `--fp16`, **talker weights** (28 layers, codec head, text projection = 171 weights) are stored as FP16 on GPU, using `cublasGemmEx` with FP16 inputs, F32 accumulation, and F32 output. **Code predictor weights** (5 layers + 15 lm_heads = 45 weights) remain F32 — sub-codebook predictions are quality-sensitive and storing them as FP16 causes audible artifacts.

Both 0.6B and 1.7B models fit on the 8 GB GPU in all modes. The 1.7B F32 uses 5.9 GB, leaving headroom for KV cache and buffers. FP16 is primarily useful for running alongside other GPU workloads or on smaller VRAM GPUs.

### End-to-End Inference Timing

Wall-clock time including talker decode (GPU), code predictor (GPU), vocoder (CPU), and WAV encoding. Measured with `seed=42` for determinism.

| Text | 0.6B F32 | 0.6B FP16 | 1.7B F32 | 1.7B FP16 |
|------|----------|-----------|----------|-----------|
| Short: "Hello world." | **1.8s** | 2.2s | 2.8s | 2.2s |
| Medium: "The quick brown fox..." (63 chars) | **6.0s** | 7.4s | 7.8s | 9.0s |
| Long: Douglas Adams quote (256 chars) | **7.1s** | 7.7s | 8.4s | 9.9s |

Notes:
- 0.6B F32 is fastest overall — the FP16 CPU-side activation conversion overhead can outweigh VRAM savings
- 1.7B is ~30-40% slower than 0.6B for medium/long text
- Measured at `--tts-max-steps=50` (old default). With the new default of 200, longer text produces more steps and longer audio.
- Vocoder time is identical across models (same vocoder, same codec format)

### Long Audio Test (max_steps=200)

With `--tts-max-steps=200` (new default), the model can produce up to 16s of audio. Tested with `seed=42`, `--fp16`, texts from 28 to 721 characters. Quality analyzed via `tts_long_audio_test.py`.

**0.6B FP16:**

| Text | Chars | Steps | EOS? | Duration | CB0 Repeat | Entropy Drop | Synth Time |
|------|-------|-------|------|----------|------------|-------------|------------|
| Short | 28 | 27 | yes | 2.1s | 11.5% | +0.02b | 7.5s |
| Medium | 102 | 80 | yes | 6.4s | 6.3% | +0.15b | 15.6s |
| Long | 262 | 191 | yes | 15.3s | 6.3% | -0.02b | 31.0s |
| Very long | 559 | 200 | no | 16.0s | 9.5% | +0.11b | 29.1s |
| Extra long | 721 | 200 | no | 16.0s | 6.0% | +0.09b | 28.6s |

**1.7B FP16:**

| Text | Chars | Steps | EOS? | Duration | CB0 Repeat | Entropy Drop | Synth Time |
|------|-------|-------|------|----------|------------|-------------|------------|
| Short | 28 | 25 | yes | 2.0s | 4.2% | +0.04b | 6.7s |
| Medium | 102 | 77 | yes | 6.1s | 6.6% | +0.22b | 17.7s |
| Long | 262 | 200 | no | 16.0s | 7.0% | -0.03b | 34.9s |
| Very long | 559 | 200 | no | 16.0s | 9.5% | +0.16b | 35.6s |
| Extra long | 721 | 200 | no | 16.0s | 5.0% | -0.03b | 34.5s |

Key observations:
- Both models produce natural EOS for short/medium text (no forced truncation)
- 0.6B produces 191 steps (15.3s) for 262-char text before EOS — nearly fills the 200-step buffer
- No entropy collapse or repetition spikes at 200 steps — no degeneration detected
- CB0 repeat rate stays below 10% for all cases (healthy range)
- Entropy drop between first/second half is <0.25 bits (negligible)
- Long text (500+ chars) hits the 200-step limit at ~16s audio. For the full text, the decode continues speaking until truncated.

## Quality Comparison (0.6B vs 1.7B)

Subjective listening comparison across 5 text types (greeting, narrative, technical, expressive, question) using default voice (no speaker embedding), `seed=42`, `--fp16`.

| Text type | 0.6B | 1.7B |
|-----------|------|------|
| Greeting ("Hello world") | Good | Good |
| Narrative (lighthouse keeper, 170 chars) | Good | Background noise/static |
| Technical (neural network description) | Good | Background noise/static |
| Expressive (exclamations, questions) | Good | No noise, but reverb-like room ambience |
| Question (philosophical) | Good | Good |

The 1.7B noise is seed-dependent: seeds 42 and 999 produced audible noise, while seeds 7 and 123 were clean. The 0.6B model was clean on all tested seeds.

### Codec Token Analysis

To determine whether the noise is a code bug or model characteristic, codec tokens were dumped and compared across seeds and models using `TTS_DUMP_CODES` env var (added to `tts_native.c`).

| Metric | 1.7B noisy (seed 42) | 1.7B clean (seed 7) | 0.6B clean (seed 42) |
|--------|---------------------|---------------------|---------------------|
| CB0 mean / std | 939 / 657 | 1069 / 611 | 1075 / 662 |
| Sub-code mean / std | 928 / 578 | 936 / 581 | 942 / 588 |
| Per-group entropy | 5.2-5.6 bits | 5.5-5.6 bits | 5.6 bits |
| Unique CB0 tokens | 49/50 | 49/50 | 46/50 |
| Anomalies (stuck/max-val) | None | None | None |

All token distributions are statistically indistinguishable. No anomalous patterns in noisy seeds.

Additionally, the `small_to_mtp_projection` math was verified with numpy against the safetensors weights (`tools/verify_mtp_projection.py`). All tensor shapes, projection dimensions, and numerical results match expected values.

### Conclusion

The noise is a **model characteristic**, not an implementation bug:
- MTP projection math verified correct (weight [1024, 2048] + bias [1024])
- All tensor shapes match (CP codec_embed [2048, 2048] in talker space, lm_head [2048, 1024] in CP space)
- F32 mode produces the same noise (not an FP16 artifact)
- Codec token distributions are identical between noisy/clean seeds
- Noise is seed-dependent (2/4 seeds), suggesting certain token *sequences* cause the shared vocoder to render reverb-like artifacts

### Debugging Tools

- `TTS_DUMP_CODES=<path>` env var: dumps codec tokens to a TSV file (one row per step, 16 tab-separated values)
- `tools/verify_mtp_projection.py`: verifies MTP projection shapes and math using numpy + safetensors (no PyTorch)
- `tools/compare_codec_tokens.py`: compares codec token distributions across dump files (entropy, anomaly checks, pairwise overlap)

## Future Work

- **In-context voice cloning** (`x_vector_only_mode=False`): Prepend reference audio codec tokens and transcript to the input sequence for higher quality cloning. Currently only x-vector mode is implemented.
- **1.7B quality investigation**: The 1.7B Base model produces occasional reverb/noise on some seeds. Could investigate: lower temperature sampling, different top-k values, or the Instruct model variant which may have better quality tuning.
- **Instruct model variants**: Qwen also provides `0.6B-Instruct` and `1.7B-Instruct` models which may have different quality characteristics. These use a chat-template prompt format and could be worth testing.
- **Even longer audio**: Default raised from 50 to 200 steps (~16s). No degeneration detected at 200 steps. Could push further (KV cache and buffers auto-grow), but quality at 300+ steps is untested.
- **Vocoder GPU acceleration**: The vocoder currently runs on CPU (OpenBLAS). Moving convolutions to GPU (cuBLAS or custom CUDA kernels) could significantly reduce total inference time, especially for longer audio.
- **Chunk-based vocoder streaming**: The current SSE streaming (see below) sends progress events during talker decode but runs the vocoder as a single batch, delivering the complete audio at the end. True audio streaming would vocode every N steps (e.g. 25 steps = 2s audio) while the talker continues decoding. This is architecturally feasible — the entire vocoder pipeline is causal (causal convolutions, causal attention in pre-transformer) — but requires:
  - KV cache for the 8-layer pre-transformer (~13 MB, persisted across chunks per layer)
  - Ring buffers for causal convolution state at each boundary (~1-2 MB total)
  - ConvTranspose1d boundary stitching for upsample/BigVGAN stages
  - Reworked buffer management (chunk-sized vs full-sequence)

  Runtime cost is neutral: total FLOPs are identical, GEMM efficiency drops ~0-5% from smaller batch sizes, and peak memory actually decreases (~50 MB vs ~280 MB) because BigVGAN working buffers scale with chunk size instead of full T. Implementation cost is ~2000 lines across vocoder files.

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `tts_pipeline.c` | ~290 | Pipeline orchestration, WAV encoding |
| `tts_native.c` | ~1520 | Native talker LM + code predictor (C + cuBLAS) |
| `tts_vocoder.c` | ~870 | Vocoder pipeline, weight loading, RVQ decode, buffer management |
| `tts_vocoder_ops.c` | ~230 | Conv1d, ConvTranspose1d, SnakeBeta, LayerNorm, GELU |
| `tts_vocoder_xfmr.c` | ~240 | 8-layer pre-transformer |
| `tts_vocoder.h` | ~200 | Vocoder types, constants, weight structs |
| `tts_sampling.c` | ~115 | Top-k sampling, repetition penalty |

## Voice Cloning

Both Base model variants (0.6B-Base and 1.7B-Base) include 76 `speaker_encoder.*` tensors (ECAPA-TDNN) alongside the talker and code predictor weights. Voice cloning uses precomputed speaker embeddings stored in `voice_presets.bin`. The embed dim is 1024 (0.6B) or 2048 (1.7B) — preset files are model-specific.

### How It Works

The [qwen-tts Python package](https://github.com/QwenLM/Qwen3-TTS) supports two voice cloning modes:

1. **X-vector only** (`x_vector_only_mode=True`): Reference audio is processed through the speaker encoder to produce a speaker embedding (1024-dim for 0.6B, 2048-dim for 1.7B). This embedding is inserted as a single extra token in the talker LM's input sequence. No reference text needed. **This is the mode we implement.**

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
FC: Conv1d(3072->embed_dim, k=1) -> speaker embedding (1024 or 2048 dim)
```

### Speaker Embedding Injection

The speaker embedding is inserted as a single token between the two parts of the codec prefix in `build_prefill_embeddings()`:

```
Without speaker:  [think, think_bos, lang_id, think_eos, pad, bos]
With speaker:     [think, think_bos, lang_id, think_eos, SPEAKER, pad, bos]
```

Text side pairs the speaker position with a `tts_pad` embedding. Prefill length increases by 1.

### Voice Presets

Precomputed speaker embeddings stored in `<tts-model-dir>/voice_presets.bin`. Embed dim is auto-detected from file size (1024 for 0.6B, 2048 for 1.7B). Preset files are model-specific and must match the loaded model's embed dim. The `voice` parameter in the API selects a preset by name.

Generate presets from reference WAV files:
```bash
C:/Data/R/git/claude-repos/local-ai-server/build.bat presets
bin/voice_presets --model <tts-model-dir> --samples voice_samples/ --output voice_presets.bin
```
