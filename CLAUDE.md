# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenAI-compatible local inference server for speech recognition (ASR) and text-to-speech (TTS), written in pure C with optional GPU acceleration. Cross-platform (Windows + Linux), single-threaded, designed for personal/local use.

- **ASR:** Qwen3-ASR (0.6B / 1.7B) via `qwen-asr` git submodule
- **TTS:** Qwen3-TTS (0.6B) with native C+cuBLAS decode and native C vocoder

## Build

### Windows (MSVC)

Requires MSVC (Visual Studio C++ workload). Set `DEPS_ROOT` environment variable pointing to dependency directory.

```bash
git submodule update --init
C:/Data/R/git/claude-repos/local-ai-server/build.bat           # auto-detects OpenBLAS, CUDA
C:/Data/R/git/claude-repos/local-ai-server/build.bat bench     # vocoder benchmark
```

**Invoking `.bat` from Claude Code (git bash):** Use the full absolute path with forward slashes (e.g. `C:/Data/R/git/claude-repos/local-ai-server/build.bat`). Git bash delegates `.bat` files to the Windows command processor automatically. Do NOT wrap in `cmd.exe /c` or `cmd //c` — this causes quoting and environment problems.

**Output:** `bin/local-ai-server.exe`

**Dependency locations** (auto-detected by build.bat):
- OpenBLAS: `%DEPS_ROOT%/openblas/`
- CUDA 12.x: system install (for cuBLAS GPU acceleration)

### Linux (gcc / Makefile)

Requires gcc and libopenblas-dev. No CUDA support yet.

```bash
git submodule update --init
sudo apt-get install libopenblas-dev   # Ubuntu/Debian
make blas       # optimized build with OpenBLAS
make debug      # debug build with AddressSanitizer
make clean      # remove build artifacts
make info       # show build configuration
```

**Output:** `bin/local-ai-server`

The Makefile also supports macOS (uses `-framework Accelerate` instead of OpenBLAS).

### Compilation tiers (both platforms)

- Inference code (`qwen-asr/`, vocoder): `-O2 -march=native -ffast-math` (gcc) / `/O2 /arch:AVX2 /fp:fast` (MSVC)
- Server code: `-g -DDEBUG` (gcc) / `/Od /Zi` (MSVC)
- Conditional defines: `USE_BLAS`, `USE_CUBLAS`, `USE_CUDA_KERNELS`

## Running

```bash
bin/local-ai-server.exe \
  --model=path/to/qwen3-asr-0.6b \
  --tts-model=path/to/qwen3-tts-12hz-0.6b-base \
  --port=8090 --threads=4 --verbose
```

TTS auto-locates vocoder weights as sibling directory: `<tts-model>/../Qwen3-TTS-Tokenizer-12Hz/model.safetensors`

## Testing

No automated test suite in the main repo. Testing approaches:

```bash
# Health check
curl http://localhost:8090/health

# ASR test
curl -X POST http://localhost:8090/v1/audio/transcriptions -F "file=@test.wav"

# TTS test (deterministic with seed)
curl -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","voice":"alloy","seed":42}'
```

### TTS Regression (`tts_regression.py`)

Automated regression harness comparing TTS output against reference WAVs using Pearson correlation and SNR metrics. Requires `seed` parameter for deterministic sampling.

```bash
python tts_regression.py --generate-missing   # First-time: generate reference WAVs
python tts_regression.py                       # Run regression checks
python tts_regression.py --sanity-only         # Non-silence + duration checks only
python tts_regression.py --refresh-refs        # Regenerate all references
python tts_regression.py --case short_hello    # Run specific case
```

Reference WAVs stored in `tts_samples/` (tracked in git).

The `qwen-asr` submodule has its own regression suite (`asr_regression.py`).

## Architecture

### Request Flow

`main.c` (CLI parsing, model loading) → `http.c` (socket accept loop) → handler dispatch:
- `handler_asr.c` routes `/v1/audio/transcriptions`, `/v1/models`, `/health`
- `handler_tts.c` routes `/v1/audio/speech`

### TTS Pipeline (`tts_pipeline.c`)

Text → tokenize → talker LM (GPU, `tts_native.c`) → code predictor (GPU) → codec tokens → vocoder (CPU, `tts_vocoder.c`) → WAV

Key TTS source files:
- `tts_native.c` — Native C+cuBLAS talker LM and code predictor (~1900 lines)
- `tts_vocoder.c/.h` — Vocoder orchestration (RVQ decode, pre-transformer, BigVGAN)
- `tts_vocoder_ops.c` — Low-level ops: convolution, SnakeBeta, LayerNorm, GELU
- `tts_vocoder_xfmr.c` — 8-layer pre-transformer with RoPE and RMSNorm
- `tts_sampling.c` — Top-k sampling, repetition penalty

### API Endpoints

| Method | Path | Handler |
|--------|------|---------|
| POST | `/v1/audio/transcriptions` | `handler_asr.c` — multipart/form-data |
| POST | `/v1/audio/speech` | `handler_tts.c` — JSON body (`input`, `voice`, `language`, `temperature`, `top_k`, `speed`, `seed`). `seed` forces single-threaded inference for determinism. |
| GET | `/v1/models` | `handler_asr.c` |
| GET | `/health` | `handler_asr.c` |

Compatible with the OpenAI Python client (`base_url="http://localhost:8090/v1"`).

### Supporting Modules

- `multipart.c` — Zero-copy binary-safe multipart parser
- `json.c` — Buffer-based JSON writer
- `json_reader.c` — JSON field extractor for request parsing

### ASR Submodule (`qwen-asr/`)

Standalone Qwen3-ASR inference engine. See `qwen-asr/AGENT.md` for its detailed implementation guide. Key headers reused by the main server: `qwen_asr.h`, `qwen_asr_kernels.h`, `qwen_asr_tokenizer.h`, `qwen_asr_safetensors.h`.

## Code Conventions

- **Flat C namespace** — all functions prefixed with module name (`http_`, `tts_`, `qwen_`, `json_`)
- **Explicit memory management** — malloc/free, preallocated scratch buffers for hot paths
- **Weight loading** — safetensors via mmap, BF16→F32 conversion on-demand
- **No external JSON/HTTP libraries** — everything is hand-rolled for minimal dependencies

### Cross-platform portability

- **Timing** — use `platform_time_ms()` from `platform.h` (QPC on Windows, `clock_gettime` on Linux). Never use `QueryPerformanceCounter` or `<windows.h>` directly in source files.
- **Sockets** — `http.h` provides `SOCKET`, `INVALID_SOCKET`, `closesocket` on both platforms. Use `SOCK_ERRNO` (in `http.c`) instead of `WSAGetLastError()`.
- **String compat** — use `strdup` (not `_strdup`), `strncasecmp`-compatible via macros in `http.c`. Add `#ifdef _MSC_VER` / `#define strdup _strdup` when needed (see `handler_asr.c` pattern).
- **Signal handling** — `main.c` uses `SetConsoleCtrlHandler` on Windows, `signal(SIGINT/SIGTERM)` on Linux.
- **Conditional features** — CUDA (`USE_CUBLAS`) is `#ifdef`-guarded and compiles cleanly without. Linux builds currently don't have CUDA.

## Key Documentation

- `QWEN3-TTS.md` — TTS architecture, model requirements, pipeline stages, voice cloning plan
- `VOCODER_OPT.md` — Vocoder optimization log (39x speedup breakdown with per-stage timing)
- `qwen-asr/AGENT.md` — ASR submodule implementation guide, behavioral contracts, regression workflow
- `tools/README.md` — Vocoder debugging and verification scripts
