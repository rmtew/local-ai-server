#!/bin/bash
# Download Qwen3-TTS 0.6B models for native inference.
# Usage: bash tools/download_tts_models.sh
#
# Downloads:
#   $DEPS_ROOT/models/tts/qwen3-tts-0.6b/
#     model.safetensors          (1.9 GB) -- talker + code predictor weights
#     vocab.json, merges.txt, config.json, tokenizer_config.json
#   $DEPS_ROOT/models/tts/Qwen3-TTS-Tokenizer-12Hz/
#     model.safetensors          (682 MB) -- vocoder weights

set -e

if [ -z "$DEPS_ROOT" ]; then
    echo "ERROR: DEPS_ROOT not set"
    exit 1
fi

MODEL_DIR="$DEPS_ROOT/models/tts/qwen3-tts-0.6b"
VOCODER_DIR="$DEPS_ROOT/models/tts/Qwen3-TTS-Tokenizer-12Hz"
mkdir -p "$MODEL_DIR" "$VOCODER_DIR"

download() {
    local url="$1"
    local dest="$2"
    local label="$3"
    if [ -f "$dest" ]; then
        echo "  SKIP $label (already exists)"
    else
        echo "  GET  $label ..."
        curl -fL -o "$dest" "$url" --progress-bar
    fi
}

echo "=== Qwen3-TTS model download ==="
echo ""

# --- Talker + code predictor weights (safetensors, native C+cuBLAS) ---
echo "[1/3] Talker + code predictor weights (1.9 GB)"
download \
    "https://huggingface.co/Qwen/Qwen3-TTS-0.6B/resolve/main/model.safetensors" \
    "$MODEL_DIR/model.safetensors" \
    "model.safetensors (Qwen3-TTS-0.6B)"

echo ""

# --- Vocoder weights (safetensors, native C) ---
echo "[2/3] Vocoder weights (682 MB)"
download \
    "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/model.safetensors" \
    "$VOCODER_DIR/model.safetensors" \
    "model.safetensors (Qwen3-TTS-Tokenizer-12Hz)"

echo ""

# --- Tokenizer files ---
echo "[3/3] Tokenizer files"
TOK_BASE="https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL/resolve/main/models/Qwen3-TTS-12Hz-0.6B-Base"
download "$TOK_BASE/vocab.json"            "$MODEL_DIR/vocab.json"            "vocab.json"
download "$TOK_BASE/merges.txt"            "$MODEL_DIR/merges.txt"            "merges.txt"
download "$TOK_BASE/config.json"           "$MODEL_DIR/config.json"           "config.json"
download "$TOK_BASE/tokenizer_config.json" "$MODEL_DIR/tokenizer_config.json" "tokenizer_config.json"

echo ""
echo "=== Download complete ==="
echo ""
echo "Model directory:"
ls -lh "$MODEL_DIR/"
echo ""
echo "Vocoder directory:"
ls -lh "$VOCODER_DIR/"
echo ""
echo "Start server with:"
echo "  bin/local-ai-server.exe --tts-model=$MODEL_DIR --verbose"
