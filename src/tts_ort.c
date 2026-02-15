/*
 * tts_ort.c - ONNX Runtime initialization for Qwen3-TTS
 *
 * The full TTS pipeline (talker, code predictor, vocoder, speaker encoder)
 * is now native C. ONNX Runtime is kept only for legacy compatibility.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "tts_ort.h"

#ifdef USE_ORT

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnxruntime_c_api.h"

/* Check ORT status, print error and return -1 on failure */
#define ORT_CHECK(api, status) do { \
    if (status) { \
        const char *msg = (api)->GetErrorMessage(status); \
        fprintf(stderr, "ORT error: %s\n", msg); \
        (api)->ReleaseStatus(status); \
        return -1; \
    } \
} while (0)

/* Load a single ONNX session from model_dir/filename.
 * Returns 0 on success, -1 on failure. session may be NULL on skip. */
static int load_session(TtsOrt *ort, const char *model_dir,
                        const char *filename, OrtSession **out, int required) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", model_dir, filename);

    /* Check file exists */
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (required) {
            fprintf(stderr, "TTS: required model not found: %s\n", path);
            return -1;
        }
        if (ort->verbose) {
            printf("  TTS: optional model not found: %s (skipped)\n", path);
        }
        *out = NULL;
        return 0;
    }
    fclose(f);

    if (ort->verbose) {
        printf("  TTS: loading %s...\n", filename);
    }

#ifdef _WIN32
    /* Convert to wide string for ORT API (Windows requires wchar_t path) */
    wchar_t wpath[512];
    size_t len = strlen(path);
    for (size_t i = 0; i <= len; i++) {
        wpath[i] = (wchar_t)(path[i] == '/' ? '\\' : path[i]);
    }
    OrtStatus *status = ort->api->CreateSession(ort->env, wpath, ort->opts, out);
#else
    OrtStatus *status = ort->api->CreateSession(ort->env, path, ort->opts, out);
#endif
    ORT_CHECK(ort->api, status);

    return 0;
}

int tts_ort_init(TtsOrt *ort, const char *model_dir, int verbose) {
    memset(ort, 0, sizeof(*ort));
    ort->verbose = verbose;

    /* Get the ORT API */
    const OrtApiBase *api_base = OrtGetApiBase();
    ort->api = api_base->GetApi(ORT_API_VERSION);
    if (!ort->api) {
        fprintf(stderr, "TTS: failed to get ORT API v%d\n", ORT_API_VERSION);
        return -1;
    }

    /* Create environment */
    OrtStatus *status;
    /* Use WARNING level to suppress noisy GetCpuPreferredNodes INFO spam */
    status = ort->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "tts", &ort->env);
    ORT_CHECK(ort->api, status);

    /* Create session options */
    status = ort->api->CreateSessionOptions(&ort->opts);
    ORT_CHECK(ort->api, status);

    ort->api->SetSessionGraphOptimizationLevel(ort->opts, ORT_ENABLE_ALL);
    ort->api->SetIntraOpNumThreads(ort->opts, 0); /* 0 = use all cores */

    /* Create CPU memory info */
    status = ort->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort->mem_info);
    ORT_CHECK(ort->api, status);

    /* Speaker encoder ONNX (unused, native C implementation in tts_speaker_enc.c) */
    if (load_session(ort, model_dir, "speaker_encoder.onnx", &ort->speaker_encoder, 0) != 0) return -1;

    return 0;
}

void tts_ort_free(TtsOrt *ort) {
    if (!ort->api) return;

    if (ort->speaker_encoder) ort->api->ReleaseSession(ort->speaker_encoder);

    if (ort->mem_info) ort->api->ReleaseMemoryInfo(ort->mem_info);
    if (ort->opts) ort->api->ReleaseSessionOptions(ort->opts);
    if (ort->env) ort->api->ReleaseEnv(ort->env);

    memset(ort, 0, sizeof(*ort));
}

#endif /* USE_ORT */
