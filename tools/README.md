# tools/

Scripts for model downloading and vocoder development/debugging.

## Download Scripts

| Script | Purpose |
|--------|---------|
| `download_tts_models.sh` | Download Qwen3-TTS models and tokenizer files from HuggingFace |

## Vocoder Verification Scripts

These Python scripts verify the native C vocoder against reference implementations. They require `numpy`.

All scripts expect to be run from the repository root and reference model files via hardcoded paths. They operate on intermediate dump files (`.raw`) produced by the native vocoder when running in verbose mode.

### Weight Verification

| Script | Purpose |
|--------|---------|
| `compare_weights2.py` | Deep comparison: extract matching weights from two safetensors files |
| `compare_weights3.py` | Match weights by value (for unnamed weights) |
| `check_shapes.py` | Print weight shapes for upsample and BigVGAN layers |

### Audio Comparison

| Script | Purpose |
|--------|---------|
| `compare_audio.py` | Compare final audio output: correlation, SNR, per-segment analysis |
| `analyze_audio.py` | Analyze audio differences (cross-correlation, linear fit) |

### Stage Verification

| Script | Purpose |
|--------|---------|
| `verify_xfmr.py` | Run Python transformer on native input and compare output |
| `verify_rvq.py` | Verify RVQ decode against native C implementation |
| `verify_stages.py` | Verify vocoder stages against native C |
| `verify_vocoder.py` | End-to-end vocoder verification |

### Voice Preset Generation

| Script | Purpose |
|--------|---------|
| `generate_voice_presets.py` | Generate voice_presets.bin from reference WAV files |
