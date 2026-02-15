/*
 * tts_pipeline.c - Qwen3-TTS synthesis pipeline
 *
 * Implements the full TTS pipeline:
 *   1. BPE tokenize input text
 *   2. Build embeddings (text_project + codec_embed via ONNX)
 *   3. Native C+cuBLAS talker decode + code predictor
 *   4. Native C vocoder -> audio
 *   5. WAV encoding
 */

#define _CRT_SECURE_NO_WARNINGS
#include "tts_pipeline.h"

#ifdef USE_ORT

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#ifdef USE_CUBLAS
#include "qwen_asr_gpu.h"
#endif

#define NUM_CODE_GROUPS     16
#define WAV_SAMPLE_RATE     24000

/* ---- WAV encoding ---- */

static void write_le16(unsigned char *p, uint16_t v) {
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
}

static void write_le32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
    p[2] = (unsigned char)((v >> 16) & 0xFF);
    p[3] = (unsigned char)((v >> 24) & 0xFF);
}

/* Encode float audio samples as 16-bit PCM WAV file.
 * speed adjusts playback rate: >1.0 = faster, <1.0 = slower.
 * Returns malloc'd WAV data, sets *out_len. */
static unsigned char *encode_wav(const float *samples, int n_samples,
                                 float speed, size_t *out_len) {
    size_t data_size = (size_t)n_samples * 2;  /* 16-bit = 2 bytes per sample */
    size_t wav_size = 44 + data_size;
    unsigned char *wav = (unsigned char *)malloc(wav_size);
    if (!wav) return NULL;

    /* Adjust effective sample rate for speed control.
     * Higher rate = player consumes samples faster = speech sounds faster. */
    uint32_t effective_rate = (uint32_t)(WAV_SAMPLE_RATE * speed + 0.5f);

    /* RIFF header */
    memcpy(wav, "RIFF", 4);
    write_le32(wav + 4, (uint32_t)(wav_size - 8));
    memcpy(wav + 8, "WAVE", 4);

    /* fmt chunk */
    memcpy(wav + 12, "fmt ", 4);
    write_le32(wav + 16, 16);                    /* chunk size */
    write_le16(wav + 20, 1);                     /* PCM format */
    write_le16(wav + 22, 1);                     /* mono */
    write_le32(wav + 24, effective_rate);         /* sample rate */
    write_le32(wav + 28, effective_rate * 2);     /* byte rate */
    write_le16(wav + 32, 2);                     /* block align */
    write_le16(wav + 34, 16);                    /* bits per sample */

    /* data chunk */
    memcpy(wav + 36, "data", 4);
    write_le32(wav + 40, (uint32_t)data_size);

    /* Convert float [-1,1] to int16 */
    int16_t *pcm = (int16_t *)(wav + 44);
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        pcm[i] = (int16_t)(s * 32767.0f);
    }

    *out_len = wav_size;
    return wav;
}

/* ---- Pipeline initialization ---- */

int tts_pipeline_init(TtsPipeline *tts, const char *model_dir, int verbose) {
    memset(tts, 0, sizeof(*tts));
    tts->verbose = verbose;

    /* Load ONNX sessions (required for vocoder at minimum) */
    if (tts_ort_init(&tts->ort, model_dir, verbose) != 0) {
        return -1;
    }

    /* Load BPE tokenizer (same Qwen vocab as ASR) */
    char vocab_path[512];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);
    tts->tokenizer = qwen_tokenizer_load(vocab_path);
    if (!tts->tokenizer) {
        fprintf(stderr, "TTS: failed to load tokenizer from %s\n", vocab_path);
        tts_ort_free(&tts->ort);
        return -1;
    }

    if (verbose) {
        printf("TTS tokenizer loaded: %d tokens\n", tts->tokenizer->vocab_size);
    }

    /* Load native C+cuBLAS decode engine (talker + code predictor from safetensors). */
    tts->native = (tts_native_ctx_t *)calloc(1, sizeof(tts_native_ctx_t));
    if (!tts->native) {
        fprintf(stderr, "TTS: failed to allocate native context\n");
        tts_ort_free(&tts->ort);
        qwen_tokenizer_free(tts->tokenizer);
        tts->tokenizer = NULL;
        return -1;
    }
    {
        void *gpu = NULL;
#ifdef USE_CUBLAS
        extern qwen_gpu_ctx_t *g_gpu_ctx;
        /* Initialize GPU context if not already created (e.g. TTS without ASR) */
        if (!g_gpu_ctx) {
            g_gpu_ctx = qwen_gpu_init();
        }
        gpu = g_gpu_ctx;
#endif
        if (tts_native_init(tts->native, model_dir, gpu, verbose) != 0) {
            fprintf(stderr, "TTS: native init failed (need model.safetensors in %s)\n", model_dir);
            free(tts->native);
            tts->native = NULL;
            tts_ort_free(&tts->ort);
            qwen_tokenizer_free(tts->tokenizer);
            tts->tokenizer = NULL;
            return -1;
        }
    }

    /* Load native vocoder from sibling Qwen3-TTS-Tokenizer-12Hz directory. */
    {
        char voc_dir[512];
        snprintf(voc_dir, sizeof(voc_dir), "%s/../Qwen3-TTS-Tokenizer-12Hz", model_dir);

        tts->vocoder = (tts_vocoder_ctx_t *)calloc(1, sizeof(tts_vocoder_ctx_t));
        if (!tts->vocoder || tts_vocoder_init(tts->vocoder, voc_dir, verbose) != 0) {
            fprintf(stderr, "TTS: native vocoder init failed (need %s/model.safetensors)\n", voc_dir);
            free(tts->vocoder);
            tts->vocoder = NULL;
            tts_native_free(tts->native);
            free(tts->native);
            tts->native = NULL;
            tts_ort_free(&tts->ort);
            qwen_tokenizer_free(tts->tokenizer);
            tts->tokenizer = NULL;
            return -1;
        }
        if (verbose) {
            printf("TTS: native vocoder loaded from %s\n", voc_dir);
        }
    }

    return 0;
}

void tts_pipeline_free(TtsPipeline *tts) {
    if (tts->vocoder) {
        tts_vocoder_free(tts->vocoder);
        free(tts->vocoder);
        tts->vocoder = NULL;
    }
    if (tts->native) {
        tts_native_free(tts->native);
        free(tts->native);
        tts->native = NULL;
    }
    if (tts->tokenizer) {
        qwen_tokenizer_free(tts->tokenizer);
        tts->tokenizer = NULL;
    }
    tts_ort_free(&tts->ort);
}

/* NOTE: The ONNX decode path (text_project, codec_embed, talker_prefill,
 * talker_decode, code_predictor) has been fully replaced by tts_native.c.
 * Only the ONNX infrastructure (env, session options, memory info) is kept
 * for potential future use (e.g. speaker_encoder for voice cloning). */


/* ---- Main synthesis function ---- */

int tts_pipeline_synthesize(TtsPipeline *tts, const char *text,
                            float temperature, int top_k, float speed,
                            TtsResult *result) {
    memset(result, 0, sizeof(*result));

    /* Clamp speed to reasonable range */
    if (speed < 0.25f) speed = 0.25f;
    if (speed > 4.0f) speed = 4.0f;

    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start);

    if (tts->verbose) {
        printf("TTS synthesize: \"%s\"\n", text);
    }

    /* Native C+cuBLAS decode (talker + code predictor) */
    int n_steps = 0;
    int64_t *codes = NULL;
    int rc = tts_native_decode(tts->native, text, temperature, top_k,
                                &codes, &n_steps);
    if (rc != 0 || !codes || n_steps == 0) {
        free(codes);
        fprintf(stderr, "TTS: native decode failed\n");
        return -1;
    }

    if (tts->verbose) {
        printf("  decode: %d steps (%.1f seconds of audio)\n",
               n_steps, n_steps * 0.08);
    }

    /* 4. Vocoder decode: codes -> audio */
    LARGE_INTEGER t_voc_start;
    QueryPerformanceCounter(&t_voc_start);
    double decode_ms = (double)(t_voc_start.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;
    if (tts->verbose) {
        printf("  vocoder: starting (%d steps, %d code groups)... [decode took %.0f ms]\n",
               n_steps, NUM_CODE_GROUPS, decode_ms);
        fflush(stdout);
    }
    int n_samples = 0;
    float *audio = tts_vocoder_run(tts->vocoder, codes, n_steps, &n_samples, NULL);
    free(codes);

    if (tts->verbose) {
        LARGE_INTEGER t_voc_end;
        QueryPerformanceCounter(&t_voc_end);
        double voc_ms = (double)(t_voc_end.QuadPart - t_voc_start.QuadPart) * 1000.0 / (double)freq.QuadPart;
        printf("  vocoder: took %.0f ms\n", voc_ms);
    }

    if (!audio || n_samples == 0) {
        free(audio);
        fprintf(stderr, "TTS: vocoder produced no audio\n");
        return -1;
    }

    if (tts->verbose) {
        printf("  vocoder: %d samples (%.1f seconds at %d Hz)\n",
               n_samples, (double)n_samples / WAV_SAMPLE_RATE, WAV_SAMPLE_RATE);
    }

    /* 5. Encode as WAV */
    size_t wav_len = 0;
    unsigned char *wav = encode_wav(audio, n_samples, speed, &wav_len);
    free(audio);

    if (!wav) {
        fprintf(stderr, "TTS: WAV encoding failed\n");
        return -1;
    }

    QueryPerformanceCounter(&t_end);
    double elapsed_ms = (double)(t_end.QuadPart - t_start.QuadPart) * 1000.0 / (double)freq.QuadPart;

    result->wav_data = wav;
    result->wav_len = wav_len;
    result->n_samples = n_samples;
    result->n_steps = n_steps;
    result->elapsed_ms = elapsed_ms;

    if (tts->verbose) {
        printf("  TTS complete: %d steps, %d samples, %.0f ms\n",
               n_steps, n_samples, elapsed_ms);
    }

    return 0;
}

#endif /* USE_ORT */
