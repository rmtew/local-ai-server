# tools/

Scripts and native tools for model downloading, vocoder debugging, voice preset generation, and regression testing.

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
| `generate_voice_presets.py` | Generate voice_presets.bin from reference WAV files (Python, requires numpy) |

## Regression & Integration Tests

Run from the repository root. Require a running `local-ai-server` with `--tts-model` loaded.

| Script | Purpose |
|--------|---------|
| `tts_regression.py` | TTS regression harness: compare output WAVs against references (correlation, SNR). Supports `--stream` for SSE streaming regression |
| `test_tts_streaming.py` | Standalone SSE streaming protocol test: validates event structure, ordering, base64 audio decode, and byte-identity with non-streaming output |
| `tts_long_audio_test.py` | Long audio quality analysis: degeneration, entropy, repetition checks at high step counts |

## Native Tools

Built with `build.bat <target>`. Output in `bin/`.

### vocoder-bench (`build.bat bench`)

Standalone vocoder benchmark. Runs the vocoder on saved codec tokens, reports per-stage timing, and compares output against references.

### voice-presets (`build.bat presets`)

Self-contained voice preset generation tool. Replaces the Python toolchain for computing speaker embeddings.

**Subcommands:**

| Command | Purpose |
|---------|---------|
| `extract` | Pull clips from media files via ffmpeg |
| `generate` | Compute speaker embeddings and write `voice_presets.bin` |
| `list` | Dump contents of an existing `voice_presets.bin` |

**Examples:**

```bash
# Extract clips from a video
voice-presets extract --input=movie.mp4 --name=Chelsie \
    --timestamps="1:23-1:45, 3:10-3:22" --output-dir=voice_samples/

# Generate presets with quality analysis
voice-presets generate --model=/path/to/qwen3-tts-0.6b-base \
    --samples=voice_samples/ --verbose

# Generate with round-trip verification (loads full TTS pipeline)
voice-presets generate --model=/path/to/qwen3-tts-0.6b-base \
    --samples=voice_samples/ --roundtrip

# List existing presets
voice-presets list --presets=/path/to/voice_presets.bin
```

The `generate` command automatically prunes outlier clips (cosine similarity < 0.75 to centroid), checks convergence via leave-one-out stability, and reports quality metrics. With `--roundtrip`, it synthesizes a test sentence with each voice and measures how well the output embedding matches the input.
