/*
 * tts_pipeline.h - Qwen3-TTS synthesis pipeline
 *
 * Tokenizes text, runs the full TTS model pipeline, and produces
 * 24kHz 16-bit mono PCM audio.
 */

#ifndef LOCAL_AI_TTS_PIPELINE_H
#define LOCAL_AI_TTS_PIPELINE_H

#ifdef USE_ORT

#include "tts_ort.h"
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
    TtsOrt ort;
    qwen_tokenizer_t *tokenizer;
    int verbose;
} TtsPipeline;

/* Initialize the TTS pipeline. model_dir should contain both ONNX models
 * and tokenizer files (vocab.json, merges.txt).
 * Returns 0 on success. */
int tts_pipeline_init(TtsPipeline *tts, const char *model_dir, int verbose);

/* Free all pipeline resources. */
void tts_pipeline_free(TtsPipeline *tts);

/* Synthesize speech from text.
 * temperature: sampling temperature (0.9 default)
 * top_k: top-k sampling (50 default)
 * speed: playback speed multiplier (1.0 default, not yet implemented)
 *
 * Returns 0 on success with result filled in.
 * Caller must free result->wav_data. */
int tts_pipeline_synthesize(TtsPipeline *tts, const char *text,
                            float temperature, int top_k, float speed,
                            TtsResult *result);

#endif /* USE_ORT */
#endif /* LOCAL_AI_TTS_PIPELINE_H */
