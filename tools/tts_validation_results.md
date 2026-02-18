# TTS Validation Results — 2026-02-19

## Summary

| Model      | Quant | VRAM    | Determ | Sanity | ASR-RT | Stream | Voices |
|------------|-------|---------|--------|--------|--------|--------|--------|
| 0.6B-CV    | F32   | 3732 MB | PASS   | PASS   | PASS   | PASS   | 4/4    |
| 0.6B-CV    | FP16  | 2874 MB | PASS   | PASS   | PASS   | PASS   | 4/4    |
| 0.6B-CV    | INT8  | 2450 MB | PASS   | PASS   | PASS   | PASS   | 4/4    |
| 0.6B-Base  | F32   | 3732 MB | PASS   | PASS   | PASS   | PASS   | 1/1    |
| 0.6B-Base  | FP16  | 2874 MB | PASS   | **FAIL** | **FAIL** | PASS | 0/1  |
| 0.6B-Base  | INT8  | 2450 MB | PASS   | PASS   | PASS   | PASS   | 1/1    |
| 1.7B-Base  | F32   | 7448 MB | PASS   | PASS   | PASS   | PASS   | 1/1    |
| 1.7B-Base  | FP16  | 4732 MB | PASS   | PASS   | PASS   | PASS   | 1/1    |
| 1.7B-Base  | INT8  | 3383 MB | PASS   | PASS   | PASS   | PASS   | 1/1    |

**Total: 8/9 combos passed, 1 failure**

## Failure Details

### 0.6B-Base / FP16 — sanity + ASR round-trip

- **sanity**: too long (15.98s) — hit 200-step max decode limit
- **asr-rt**: empty ASR text — the max-steps output was garbled/repetitive

This is a model behavior issue with seed=42 + "The quick brown fox..." on 0.6B-Base FP16.
Determinism still passes (same bad output both times), and streaming works fine
(streaming uses the shorter "Hello, world." text which doesn't trigger the issue).

F32 and INT8 modes for the same model produce normal-length output with the same seed/text.

## VRAM Progression

| Model     | F32     | FP16    | INT8    |
|-----------|---------|---------|---------|
| 0.6B-CV   | 3732 MB | 2874 MB | 2450 MB |
| 0.6B-Base | 3732 MB | 2874 MB | 2450 MB |
| 1.7B-Base | 7448 MB | 4732 MB | 3383 MB |

Note: VRAM includes both TTS + ASR 0.6B model (for round-trip test).

## Test Environment

- CUDA 13.1 Update 1
- ASR model: qwen3-asr-0.6b (for round-trip tests)
- Seed: 42, Temperature: 0.3
- CustomVoice voices tested: alloy, serena, ryan, aiden
- Base voices tested: alloy
