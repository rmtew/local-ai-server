# Native C+cuBLAS TTS (Qwen3-TTS)

Replace ONNX Runtime TTS inference with native C implementation using cuBLAS
for GPU acceleration on the RTX 3070. Same approach as the qwen-asr submodule.

## Motivation

ONNX Runtime cannot properly utilize the GPU for this model:
- CUDA EP: `GetCpuPreferredNodes` heuristic forces nodes to CPU, causing memcpy
  ping-pong. GPU was 2x SLOWER than CPU-only (58s vs 39s).
- DirectML EP (AMD iGPU): 2x slower per decode step than CPU. Vocoder OOMs.
- DirectML EP (NVIDIA dGPU): Conflicts with CUDA (ASR uses cuBLAS). Device removed.
- Vocoder OOMs on every GPU attempted (needs ~750MB+, iGPU has 495MB).
- CPU-only baseline: ~12s decode + ~50s vocoder = ~60s for a few words. Unusable.

Native C+cuBLAS avoids ORT overhead entirely. The qwen-asr decoder already runs
the same Qwen3 transformer architecture on GPU successfully.

## Architecture

Qwen3-TTS 0.6B has these components:

### Talker LM (main autoregressive model) -- 28 layers
- Identical Qwen3 transformer to qwen-asr decoder
- hidden=1024, 16 heads, 8 KV heads, head_dim=128, intermediate=3072
- vocab=3072 (codec tokens + special), RoPE theta=1M
- MRoPE sections [24, 20, 20] interleaved (may collapse for TTS)
- Weights: in `model.safetensors` (1.83 GB for 0.6B)

### Code Predictor (sub-codebook, 15 iterations per step) -- 5 layers
- Same architecture as talker but smaller (5 layers)
- hidden=1024, vocab=2048
- No persistent KV cache (runs fresh each step)
- Weights: in same `model.safetensors`

### Embeddings
- text_project: 151,936 x 2048 -> linear to 1024
- codec_embed: codebook embeddings -> 1024

### Vocoder (Qwen3-TTS-Tokenizer-12Hz decoder) -- separate model
- Based on Mimi (Kyutai/Defossez 2024), which evolved from EnCodec's SEANet backbone
- ConvNet + transformer hybrid (682 MB safetensors)
- 8 transformer layers, sliding-window attention (window=72), 16 heads, head_dim=64
- RoPE theta=10000, SiLU activation
- Upsampling: ConvTranspose1d strides [8, 5, 4, 3] + additional [2, 2] = 1920x total
- Decoder dim=1536, latent=1024, hidden=512, intermediate=1024
- 16 codebooks (1 semantic size=4096, 15 acoustic size=2048), codebook dim=512
- Frame rate: 12.5 Hz (24000 / 1920), output: 24kHz PCM audio
- Input: [n_steps, 16] codec tokens -> 24kHz PCM
- Weights: `speech_tokenizer/model.safetensors`

## Phased Plan

### Phase 1: Native talker + code predictor on GPU
Reuse qwen-asr infrastructure (safetensors loader, tokenizer, cuBLAS GEMM,
CUDA kernels for RMSNorm/RoPE/SwiGLU/attention). Implement:
- Weight loading with TTS tensor name mapping
- Talker prefill + decode forward pass (reuse decoder pattern)
- Code predictor forward pass (5-layer variant)
- Embedding layers (table lookup + linear projection)
- Sampling (top-k, repetition penalty, temperature)
- Pipeline orchestration (token building, embedding summing)

Expected: ~1,500 lines new code, ~5,000 lines reused.
This replaces the 12s decode phase. Should be <1s on RTX 3070.

### Phase 2: Native vocoder on GPU

The vocoder is based on Mimi (Kyutai), which evolved from EnCodec's SEANet backbone.
Key existing C/C++ code that can be adapted:

**encodec.cpp** (github.com/PABannier/encodec.cpp, ~227 stars):
- Full C/C++ EnCodec decoder using GGML
- Already implements: Conv1d, ConvTranspose1d, residual blocks, RVQ dequantization
- Upsampling ratios [8, 5, 4, 2] -- close to our [8, 5, 4, 3] + [2, 2]
- Uses ELU activation (we need SiLU -- trivial swap)
- Uses LSTM between upsample stages (we need transformer layers -- main delta)
- GGML has CUDA/Metal kernels for ConvTranspose1d already

**bark.cpp** (github.com/PABannier/bark.cpp):
- Full Bark TTS pipeline using encodec.cpp as vocoder
- Demonstrates LLM-generates-tokens -> codec-decodes-to-audio integration pattern

**What can be adapted from encodec.cpp:**
- ConvTranspose1d operations (GGML, including GPU kernels)
- Residual block structure (swap ELU for SiLU)
- RVQ dequantization (codebook lookup + sum)
- Model weight loading infrastructure

**What needs new implementation:**
- 8 sliding-window transformer layers (window=72, RoPE, 16 heads, head_dim=64)
  - Can borrow patterns from qwen-asr decoder (same operations, different dims)
- The [2, 2] additional upsampling stages
- Weight conversion script (safetensors -> native format)

**NOT relevant:** llama.cpp's WavTokenizer uses ConvNext + iSTFT (fundamentally
different architecture, no ConvTranspose1d at all).

**Alternative reference:** Moshi project (github.com/kyutai-labs/moshi) has a Rust
implementation of Mimi -- closest architectural match to Qwen3-TTS vocoder.

Recommended approach: Fork encodec.cpp decoder, replace LSTM with transformer
layers (using qwen-asr patterns), adjust upsample ratios. This is "assembly from
existing parts" rather than from-scratch implementation.

### Phase 3: Optimization
- Quantization (INT8/INT4 weights for decode models)
- Batched sub-code prediction
- Vocoder optimization (if native)

## Reusable from qwen-asr

| Component | File | Lines | Reuse |
|-----------|------|-------|-------|
| Safetensors loader | qwen_asr_safetensors.c | 482 | 100% |
| BPE tokenizer | qwen_asr_tokenizer.c | 658 | 100% |
| CPU math kernels | qwen_asr_kernels.c | 1,445 | ~90% |
| AVX2 kernels | qwen_asr_kernels_avx.c | 502 | ~90% |
| CUDA kernels | qwen_asr_kernels.cu | 416 | ~80% |
| cuBLAS infrastructure | qwen_asr_gpu.c | 1,090 | ~80% |

## Key Risks

1. MRoPE -- may need interleaved 3-track position encoding (vs collapsed in ASR)
2. Weight name mapping -- TTS uses different prefixes than ASR
3. Vocoder transformer layers -- need sliding-window attention with window=72
4. Vocoder GPU memory -- 682 MB weights alone, plus activation memory. ORT failed
   on every GPU, but native code has tighter memory control (no ORT overhead).
5. ConvTranspose1d on GPU -- GGML has this, but may need custom CUDA kernels for
   the specific stride/padding configurations used here.

## References

- Model: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- Tokenizer: https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz
- Code: https://github.com/QwenLM/Qwen3-TTS
- Technical report: https://arxiv.org/html/2601.15621v1
- Existing ONNX pipeline: src/tts_pipeline.c (reference for inference flow)
- qwen-asr reference: qwen-asr/ submodule (architecture pattern)
- encodec.cpp: https://github.com/PABannier/encodec.cpp (adaptable decoder code)
- bark.cpp: https://github.com/PABannier/bark.cpp (TTS pipeline with encodec)
- Moshi/Mimi (Rust): https://github.com/kyutai-labs/moshi (closest arch match)
