/*
 * tts_ort.h - ONNX Runtime wrapper for Qwen3-TTS models
 *
 * Loads all 9 ONNX sessions needed for TTS inference and provides
 * tensor creation/execution helpers.
 */

#ifndef LOCAL_AI_TTS_ORT_H
#define LOCAL_AI_TTS_ORT_H

#ifdef USE_ORT

#include <stddef.h>
#include <stdint.h>

/* Forward-declare ORT types to avoid including the full header everywhere */
typedef struct OrtApi OrtApi;
typedef struct OrtEnv OrtEnv;
typedef struct OrtSession OrtSession;
typedef struct OrtSessionOptions OrtSessionOptions;
typedef struct OrtMemoryInfo OrtMemoryInfo;
typedef struct OrtValue OrtValue;
typedef struct OrtRunOptions OrtRunOptions;

/* All 9 ONNX sessions for Qwen3-TTS */
typedef struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtSessionOptions *opts;
    OrtMemoryInfo *mem_info;

    OrtSession *text_project;        /* text token IDs -> embeddings */
    OrtSession *codec_embed;         /* codec token IDs -> embeddings */
    OrtSession *talker_prefill;      /* prefill: embeddings -> logits + KV cache */
    OrtSession *talker_decode;       /* decode: single step -> logits + updated KV */
    OrtSession *code_predictor;      /* sub-code prediction */
    OrtSession *code_predictor_embed;/* sub-code token -> embedding */
    OrtSession *speaker_encoder;     /* (Phase 2: voice cloning) */
    OrtSession *tokenizer12hz_encode;/* (Phase 2: voice cloning) */
    OrtSession *tokenizer12hz_decode;/* codec tokens -> audio waveform */

    /* Cached I/O names for sessions with fixed signatures.
     * Queried once at init; used for all subsequent Run calls.
     * These are strdup'd copies (not allocator-managed). */
    char *text_project_in;       /* single input name */
    char *text_project_out;      /* single output name */
    char *codec_embed_in;        /* single input name */
    char *codec_embed_out;       /* single output name */
    char *code_pred_in_embeds;   /* "inputs_embeds" or actual name */
    char *code_pred_in_step;     /* "generation_step" or actual name */
    char *code_pred_out;         /* "logits" or actual name */
    char *code_pred_embed_in_ids;   /* input name */
    char *code_pred_embed_in_step;  /* input name */
    char *code_pred_embed_out;      /* output name */
    char *vocoder_in;            /* "audio_codes" or actual name */
    char *vocoder_out_audio;     /* first output name */
    char *vocoder_out_lengths;   /* second output name (may be NULL) */

    int verbose;
} TtsOrt;

/* Load all ONNX sessions from model_dir. Returns 0 on success. */
int tts_ort_init(TtsOrt *ort, const char *model_dir, int verbose);

/* Free all sessions and ORT resources. */
void tts_ort_free(TtsOrt *ort);

/* Create a 1D int64 tensor from data. Caller must release with api->ReleaseValue. */
OrtValue *tts_ort_tensor_int64(TtsOrt *ort, const int64_t *data, int64_t len);

/* Create a 2D int64 tensor [1, len]. */
OrtValue *tts_ort_tensor_int64_2d(TtsOrt *ort, const int64_t *data, int64_t len);

/* Create a 3D float tensor [d0, d1, d2] from data. */
OrtValue *tts_ort_tensor_float_3d(TtsOrt *ort, const float *data,
                                   int64_t d0, int64_t d1, int64_t d2);

/* Create a 2D float tensor [d0, d1] from data. */
OrtValue *tts_ort_tensor_float_2d(TtsOrt *ort, const float *data,
                                   int64_t d0, int64_t d1);

/* Run a session with named inputs/outputs. Returns 0 on success.
 * output_values[] must be pre-initialized to NULL. */
int tts_ort_run(TtsOrt *ort, OrtSession *session,
                const char **input_names, const OrtValue *const *input_values, size_t n_inputs,
                const char **output_names, OrtValue **output_values, size_t n_outputs);

/* Get float data pointer from tensor. Returns NULL on error. */
float *tts_ort_get_float_data(TtsOrt *ort, OrtValue *value);

/* Get int64 data pointer from tensor. Returns NULL on error. */
int64_t *tts_ort_get_int64_data(TtsOrt *ort, OrtValue *value);

/* Get tensor shape info. Returns number of dims, fills shape[]. */
int tts_ort_get_shape(TtsOrt *ort, OrtValue *value, int64_t *shape, int max_dims);

/* Release an OrtValue (NULL-safe). */
void tts_ort_release_value(TtsOrt *ort, OrtValue *value);

#endif /* USE_ORT */
#endif /* LOCAL_AI_TTS_ORT_H */
