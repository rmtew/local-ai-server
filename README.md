# local-ai-server

OpenAI-compatible local inference server for speech recognition and synthesis. Pure C with optional GPU acceleration. Single-threaded, designed for personal/local use.

- **ASR**: Qwen3-ASR (0.6B / 1.7B) -- native C + cuBLAS, BF16 safetensors
- **TTS**: Qwen3-TTS (0.6B) -- native C + cuBLAS for decode, native C vocoder

See [QWEN3-TTS.md](QWEN3-TTS.md) for TTS-specific architecture and model details.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/transcriptions` | Transcribe audio (requires `--model`) |
| POST | `/v1/audio/speech` | Synthesize speech (requires `--tts-model`) |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |

## Prerequisites

- **MSVC** (Visual Studio C++ workload)
- **DEPS_ROOT** environment variable pointing to shared dependencies directory
- **OpenBLAS** at `%DEPS_ROOT%/openblas/` (strongly recommended for CPU performance)
- **CUDA 12.x** (optional, enables cuBLAS GPU acceleration)
- **ONNX Runtime** at `%DEPS_ROOT%/onnxruntime/1.23.2/` (required for TTS)

## Building

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/rmtew/local-ai-server.git
cd local-ai-server

# Or if already cloned:
git submodule update --init

# Build (auto-detects MSVC, OpenBLAS, CUDA, ONNX Runtime)
build.bat
```

Output: `bin/local-ai-server.exe` (plus any required DLLs copied to `bin/`)

The build script auto-detects available libraries and sets compile flags accordingly:
- `USE_BLAS` -- OpenBLAS found
- `USE_CUBLAS`, `USE_CUDA_KERNELS` -- CUDA toolkit + nvcc found
- `USE_ORT` -- ONNX Runtime found (required for TTS)

## Model Setup

### ASR: Qwen3-ASR

Download from HuggingFace using the included script:

```bash
# Interactive (choose small=0.6B or large=1.7B)
bash qwen-asr/download_model.sh

# Non-interactive
bash qwen-asr/download_model.sh --model small --dir path/to/model
```

Or download manually from [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) or [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B). Required files: `model.safetensors` (or sharded variants), `config.json`, `vocab.json`, `merges.txt`.

### TTS: Qwen3-TTS

Two model directories are required. See [QWEN3-TTS.md](QWEN3-TTS.md) for full details.

```bash
# Download all TTS models (safetensors + embedding ONNX + tokenizer)
bash tools/download_tts_models.sh
```

## Usage

```bash
# ASR only
bin/local-ai-server.exe --model=path/to/qwen-asr-model

# TTS only
bin/local-ai-server.exe --tts-model=path/to/qwen3-tts-0.6b

# Both ASR and TTS
bin/local-ai-server.exe --model=path/to/asr --tts-model=path/to/tts --verbose
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model=<dir>` | -- | Path to Qwen3-ASR model directory |
| `--tts-model=<dir>` | -- | Path to Qwen3-TTS model directory |
| `--port=<N>` | 8090 | HTTP listen port |
| `--language=<lang>` | auto-detect | Force ASR language for all requests |
| `--threads=<N>` | 4 | CPU threads for inference |
| `--verbose` | off | Log each request with timing stats |

At least one of `--model` or `--tts-model` must be specified.

## API Reference

### POST /v1/audio/transcriptions

Transcribe audio. Multipart/form-data, compatible with the OpenAI transcription API.

**Fields:**
- `file` (required) -- WAV audio file
- `language` -- Override language for this request (e.g., "Chinese", "English")
- `response_format` -- `"json"` (default) or `"verbose_json"` (includes word timestamps)
- `model` -- Accepted but ignored (single model server)

**Response (json):**
```json
{"text": "Hello world"}
```

**Response (verbose_json):**
```json
{
  "text": "Hello world",
  "duration": 1.234,
  "words": [
    {"word": "Hello", "start": 0.120, "end": 0.450},
    {"word": "world", "start": 0.450, "end": 1.100}
  ]
}
```

### POST /v1/audio/speech

Synthesize speech. JSON body, returns WAV audio.

**Request body:**
```json
{
  "input": "Hello, world!",
  "voice": "Chelsie",
  "speed": 1.0
}
```

- `input` (required) -- Text to synthesize
- `voice` -- Voice name (currently accepted but not used for voice selection)
- `speed` -- Playback speed multiplier (0.25 to 4.0, default 1.0)
- `seed` -- Integer seed for deterministic output (forces single-threaded inference; omit for default behavior)
- `response_format` -- `"wav"` (default)

**Response:** WAV audio (16-bit PCM, 24 kHz mono)

### GET /v1/models

```json
{"data": [{"id": "qwen-asr", "object": "model"}]}
```

### GET /health

```json
{"status": "ok"}
```

## Examples

```bash
# Health check
curl http://localhost:8090/health

# Transcribe audio
curl -X POST http://localhost:8090/v1/audio/transcriptions -F "file=@test.wav"

# Transcribe with language and timestamps
curl -X POST http://localhost:8090/v1/audio/transcriptions \
  -F "file=@test.wav" -F "language=Chinese" -F "response_format=verbose_json"

# Synthesize speech
curl -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello, world!","voice":"Chelsie"}' \
  --output speech.wav
```

### OpenAI Python client

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/v1", api_key="unused")

# Transcription
result = client.audio.transcriptions.create(model="qwen-asr", file=open("test.wav", "rb"))
print(result.text)

# Speech synthesis
response = client.audio.speech.create(model="qwen-tts", input="Hello, world!", voice="Chelsie")
response.stream_to_file("speech.wav")
```

## Architecture

```
src/
  main.c                -- Entry point, arg parsing, model load, Ctrl+C handler
  http.c/.h             -- HTTP server: listen, accept, parse, respond
  multipart.c/.h        -- Binary-safe multipart/form-data parser (zero-copy)
  handler_asr.c/.h      -- ASR request handler, JSON/SSE responses
  handler_tts.c/.h      -- TTS request handler, WAV response
  json.c/.h             -- JSON writer
  json_reader.c/.h      -- JSON field extractor (TTS request parsing)
  tts_pipeline.c/.h     -- TTS orchestration: native decode, native vocoder, WAV encoding
  tts_ort.c/.h          -- ONNX Runtime initialization (for future voice cloning)
  tts_sampling.c        -- Top-k sampling, repetition penalty
  tts_native.c/.h       -- Native C+cuBLAS talker LM + code predictor
  tts_vocoder.c         -- Native C vocoder: RVQ, convolutions, BigVGAN
  tts_vocoder_ops.c     -- Conv1d, ConvTranspose1d, SnakeBeta, LayerNorm, GELU
  tts_vocoder_xfmr.c    -- 8-layer pre-transformer (RoPE, RMSNorm, SwiGLU)
  tts_vocoder.h         -- Vocoder types and constants
qwen-asr/               -- Qwen3-ASR pure C inference (git submodule)
tools/                   -- Download scripts and debugging tools (see tools/README.md)
build.bat                -- MSVC build script (auto-detects dependencies)
```

## Supported Languages (ASR)

Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian
