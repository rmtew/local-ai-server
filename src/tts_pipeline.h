/*
 * tts_pipeline.h - Qwen3-TTS synthesis pipeline
 *
 * Tokenizes text, runs the full TTS model pipeline, and produces
 * 24kHz 16-bit mono PCM audio.
 */

#ifndef LOCAL_AI_TTS_PIPELINE_H
#define LOCAL_AI_TTS_PIPELINE_H

#include "tts_native.h"
#include "tts_vocoder.h"
#include "tts_voice_presets.h"
#include "qwen_asr_tokenizer.h"
#include <stddef.h>

/* Synthesis result */
typedef struct {
    unsigned char *wav_data;  /* Complete WAV file (header + PCM), malloc'd */
    size_t wav_len;           /* Total WAV file size in bytes */
    int n_samples;            /* Number of PCM samples (24kHz) */
    int n_steps;              /* Number of autoregressive steps */
    double elapsed_ms;        /* Wall-clock time for synthesis */
} TtsResult;

/* TTS pipeline state */
typedef struct {
    qwen_tokenizer_t *tokenizer;
    tts_native_ctx_t *native;   /* Native C+cuBLAS talker + code predictor */
    tts_vocoder_ctx_t *vocoder; /* Native C vocoder */
    tts_voice_presets_t voice_presets; /* Precomputed speaker embeddings */
    int verbose;
} TtsPipeline;

/* Initialize the TTS pipeline. model_dir should contain safetensors weights
 * and tokenizer files (vocab.json, merges.txt).
 * fp16: if non-zero, store GPU weights as FP16 (half VRAM, uses tensor cores).
 * Returns 0 on success. */
int tts_pipeline_init(TtsPipeline *tts, const char *model_dir, int fp16, int verbose);

/* Free all pipeline resources. */
void tts_pipeline_free(TtsPipeline *tts);

/* Synthesize speech from text.
 * voice: voice preset name (NULL for default voice)
 * language: language hint ("auto", "english", "chinese", etc.) or NULL for auto
 * temperature: sampling temperature (0.9 default)
 * top_k: top-k sampling (50 default)
 * progress/progress_data: optional callback for streaming progress (NULL to disable).
 *   Called with phase "decoding" per step and "vocoder" once before vocoder runs.
 *
 * Returns 0 on success with result filled in.
 * Caller must free result->wav_data. */
int tts_pipeline_synthesize(TtsPipeline *tts, const char *text,
                            const char *voice, const char *language,
                            float temperature, int top_k,
                            tts_progress_fn progress, void *progress_data,
                            TtsResult *result);

#endif /* LOCAL_AI_TTS_PIPELINE_H */
