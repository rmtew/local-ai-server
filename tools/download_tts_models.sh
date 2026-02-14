#!/bin/bash
# Download Qwen3-TTS 0.6B ONNX models and tokenizer files
# Usage: bash tools/download_tts_models.sh
#
# Downloads to: $DEPS_ROOT/models/tts/qwen3-tts-0.6b/
# Total size: ~6.5 GB (ONNX models) + ~5 MB (tokenizer files)

set -e

if [ -z "$DEPS_ROOT" ]; then
    echo "ERROR: DEPS_ROOT not set"
    exit 1
fi

MODEL_DIR="$DEPS_ROOT/models/tts/qwen3-tts-0.6b"
mkdir -p "$MODEL_DIR"

HF_BASE="https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL/resolve/main"

# ONNX models from onnx_kv_06b/ subfolder
ONNX_FILES=(
    "onnx_kv_06b/talker_prefill.onnx"
    "onnx_kv_06b/talker_decode.onnx"
    "onnx_kv_06b/text_project.onnx"
    "onnx_kv_06b/tokenizer12hz_decode.onnx"
    "onnx_kv_06b/tokenizer12hz_encode.onnx"
    "onnx_kv_06b/code_predictor.onnx"
    "onnx_kv_06b/code_predictor_embed.onnx"
    "onnx_kv_06b/speaker_encoder.onnx"
    "onnx_kv_06b/codec_embed.onnx"
)

# Tokenizer files from models/Qwen3-TTS-12Hz-0.6B-Base/
TOKENIZER_FILES=(
    "models/Qwen3-TTS-12Hz-0.6B-Base/vocab.json"
    "models/Qwen3-TTS-12Hz-0.6B-Base/merges.txt"
    "models/Qwen3-TTS-12Hz-0.6B-Base/tokenizer_config.json"
    "models/Qwen3-TTS-12Hz-0.6B-Base/config.json"
)

echo "Downloading Qwen3-TTS 0.6B models to: $MODEL_DIR"
echo ""

# Download ONNX models (flatten into model dir)
for f in "${ONNX_FILES[@]}"; do
    basename=$(basename "$f")
    dest="$MODEL_DIR/$basename"
    if [ -f "$dest" ]; then
        echo "  SKIP $basename (already exists)"
    else
        echo "  GET  $basename ..."
        curl -L -o "$dest" "$HF_BASE/$f" 2>&1 | tail -1
    fi
done

# Download tokenizer files (flatten into model dir)
for f in "${TOKENIZER_FILES[@]}"; do
    basename=$(basename "$f")
    dest="$MODEL_DIR/$basename"
    if [ -f "$dest" ]; then
        echo "  SKIP $basename (already exists)"
    else
        echo "  GET  $basename ..."
        curl -L -o "$dest" "$HF_BASE/$f" 2>&1 | tail -1
    fi
done

echo ""
echo "Download complete. Contents:"
ls -lh "$MODEL_DIR/"
echo ""
echo "Start server with:"
echo "  bin/local-ai-server.exe --tts-model=$MODEL_DIR --verbose"
