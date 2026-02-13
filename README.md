# local-ai-server

OpenAI-compatible local inference server. Currently supports ASR (automatic speech recognition) via [qwen-asr](https://github.com/rmtew/qwen-asr) (Qwen3-ASR, pure C). Planned: TTS and embeddings endpoints.

Single-threaded, designed for personal/local use.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/transcriptions` | Transcribe audio (OpenAI-compatible) |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |

## Prerequisites

- **MSVC** (Visual Studio C++ workload)
- **DEPS_ROOT** environment variable pointing to shared dependencies directory
- **OpenBLAS** at `%DEPS_ROOT%/openblas/` (strongly recommended for CPU performance)
- **CUDA 12.x** (optional, enables cuBLAS GPU acceleration)

## Building

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/rmtew/local-ai-server.git
cd local-ai-server

# Or if already cloned:
git submodule update --init

# Build (auto-detects MSVC, OpenBLAS, CUDA)
build.bat
```

Output: `bin/local-ai-server.exe`

## Usage

```bash
# Start the server (model directory is required)
bin/local-ai-server.exe --model=path/to/qwen-asr-model

# With options
bin/local-ai-server.exe --model=path/to/model --port=8090 --threads=4 --language=Chinese --verbose
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model=<dir>` | (required) | Path to qwen-asr model directory |
| `--port=<N>` | 8090 | HTTP listen port |
| `--language=<lang>` | auto-detect | Force language for all transcriptions |
| `--threads=<N>` | 4 | CPU threads for inference |
| `--verbose` | off | Log each request with timing stats |
| `--help` | | Show usage and supported languages |

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

# Basic transcription
curl -X POST http://localhost:8090/v1/audio/transcriptions -F "file=@test.wav"

# With language override
curl -X POST http://localhost:8090/v1/audio/transcriptions -F "file=@test.wav" -F "language=Chinese"

# Verbose with timestamps
curl -X POST http://localhost:8090/v1/audio/transcriptions -F "file=@test.wav" -F "response_format=verbose_json"
```

### OpenAI Python client

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/v1", api_key="unused")
result = client.audio.transcriptions.create(model="qwen-asr", file=open("test.wav", "rb"))
print(result.text)
```

## Architecture

```
src/
  main.c           -- Entry point, arg parsing, model load, Ctrl+C handler
  http.c/.h        -- HTTP server: listen, accept, parse headers/body, send responses
  multipart.c/.h   -- Binary-safe multipart/form-data parser (zero-copy)
  handler_asr.c/.h -- ASR request routing, transcription logic, JSON responses
  json.c/.h        -- JSON writer (standalone, no external deps)
qwen-asr/           -- Qwen3-ASR pure C inference (git submodule)
build.bat           -- MSVC build script
```

## Supported Languages

Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian
