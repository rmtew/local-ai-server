# tools/

Scripts for model downloading and vocoder development/debugging.

## Download Scripts

| Script | Purpose |
|--------|---------|
| `download_tts_models.sh` | Download Qwen3-TTS ONNX models and tokenizer files from HuggingFace |

## Vocoder Debugging Scripts

These Python scripts were used during development of the native C vocoder to verify correctness against the ONNX reference implementation. They require `numpy`, `onnx`, and `onnxruntime`.

All scripts expect to be run from the repository root and reference model files via hardcoded paths. They operate on intermediate dump files (`.raw`) produced by the native vocoder when running in verbose mode.

### Weight Verification

| Script | Purpose |
|--------|---------|
| `compare_onnx_weights.py` | Compare weights between ONNX model and safetensors file |
| `compare_weights2.py` | Deep comparison: extract matching weights from ONNX and safetensors |
| `compare_weights3.py` | Match ONNX-only weights to safetensors by value (for unnamed weights) |
| `check_shapes.py` | Print weight shapes for upsample and BigVGAN layers |

### Stage-by-Stage Comparison

| Script | Purpose |
|--------|---------|
| `compare_all_stages.py` | Compare native vs ONNX at every pipeline stage using same codec tokens |
| `compare_stages.py` | Compare native vs ONNX intermediate outputs from dump files |
| `compare_audio.py` | Compare final audio output: correlation, SNR, per-segment analysis |

### Transformer Verification

| Script | Purpose |
|--------|---------|
| `compare_xfmr_onnx.py` | Extract ONNX transformer intermediate outputs and compare with native |
| `compare_xfmr_three.py` | Three-way comparison: Python reference vs native C vs ONNX |
| `compare_layer0.py` | Layer 0 intermediate comparison (found the RMSNorm epsilon bug) |
| `verify_xfmr.py` | Run Python transformer on native input and compare output |

### BigVGAN Verification

| Script | Purpose |
|--------|---------|
| `compare_bigvgan.py` | Compare BigVGAN decoder stages between native and ONNX |
| `compare_bigvgan2.py` | Step-by-step BigVGAN block 0 comparison |
| `compare_bigvgan3.py` | Extract ONNX Slice parameters (found the ConvTranspose trim bug) |

### End-to-End Verification

| Script | Purpose |
|--------|---------|
| `compare_onnx_run.py` | Run ONNX vocoder standalone and extract intermediate outputs |
| `verify_rvq.py` | Verify RVQ decode against native C implementation |
| `verify_stages.py` | Verify vocoder stages against native C |
| `verify_vocoder.py` | End-to-end vocoder verification |
| `analyze_audio.py` | Analyze audio differences (cross-correlation, linear fit) |

### Key Bugs Found

These scripts were instrumental in finding two bugs in the native vocoder:

1. **RMSNorm epsilon mismatch** (`compare_layer0.py`): Native used 1e-6, ONNX uses 1e-5. Found by comparing layer 0 intermediate values and seeing 1000x amplification after normalization.

2. **ConvTranspose1d causal trim direction** (`compare_bigvgan3.py`): Native trimmed `kernel-stride` from the right; ONNX trims from both sides. Found by extracting the ONNX Slice node parameters for all four BigVGAN blocks.
