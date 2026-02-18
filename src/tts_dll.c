/*
 * tts_dll.c - DLL entry point for TTS pipeline
 *
 * Thin wrapper exporting a C API suitable for Python ctypes (or any FFI).
 * Handles opaque allocation, seed/thread management, and WAV buffer lifecycle.
 *
 * Build: build.bat ttsdll -> bin/tts_pipeline.dll
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tts_pipeline.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_gpu.h"
#include "platform.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

/* Required global (referenced by tts_native.c decode loop) */
volatile int g_shutdown = 0;

/* Track thread count for save/restore during seeded synthesis */
static int s_threads = 4;

/* ---- Exported API ---- */

DLL_EXPORT TtsPipeline *tts_dll_create(void) {
    TtsPipeline *tts = (TtsPipeline *)calloc(1, sizeof(TtsPipeline));
    return tts;
}

DLL_EXPORT void tts_dll_destroy(TtsPipeline *tts) {
    if (!tts) return;
    tts_pipeline_free(tts);
    free(tts);
}

DLL_EXPORT int tts_dll_init(TtsPipeline *tts, const char *model_dir,
                            int fp16, int int8, int verbose) {
    if (!tts || !model_dir) return -1;
    return tts_pipeline_init(tts, model_dir, fp16, int8, verbose);
}

DLL_EXPORT int tts_dll_synthesize(TtsPipeline *tts, const char *text,
                                   const char *voice, const char *language,
                                   float temperature, int top_k, int seed,
                                   int *n_steps, int *n_samples,
                                   double *elapsed_ms,
                                   unsigned char **wav_data, size_t *wav_len) {
    if (!tts || !text) return -1;

    /* Seed RNG and force single-threaded GEMM for determinism when seed >= 0,
     * matching the server's behavior (handler_tts.c). */
    int saved_threads = 0;
    if (seed >= 0) {
        srand((unsigned int)seed);
        saved_threads = s_threads;
        qwen_set_threads(1);
    } else {
        unsigned int auto_seed = (unsigned int)(platform_time_ms() * 1000.0);
        srand(auto_seed);
    }

    TtsResult result;
    int rc = tts_pipeline_synthesize(tts, text,
                                      voice, language,
                                      temperature, top_k,
                                      NULL, NULL,  /* no progress callback */
                                      &result);

    if (saved_threads > 0) {
        qwen_set_threads(saved_threads);
    }

    if (rc != 0) return rc;

    /* Return results via out-pointers */
    if (n_steps)    *n_steps    = result.n_steps;
    if (n_samples)  *n_samples  = result.n_samples;
    if (elapsed_ms) *elapsed_ms = result.elapsed_ms;
    if (wav_data)   *wav_data   = result.wav_data;
    if (wav_len)    *wav_len    = result.wav_len;

    return 0;
}

DLL_EXPORT void tts_dll_free_wav(unsigned char *wav_data) {
    free(wav_data);
}

DLL_EXPORT void tts_dll_set_threads(int n) {
    if (n >= 1) {
        s_threads = n;
        qwen_set_threads(n);
    }
}

DLL_EXPORT int tts_dll_get_vram_mb(void) {
#ifdef USE_CUBLAS
    extern qwen_gpu_ctx_t *g_gpu_ctx;
    if (g_gpu_ctx) {
        qwen_gpu_stats_t stats;
        qwen_gpu_get_stats(g_gpu_ctx, &stats);
        return (int)((stats.vram_weights + stats.vram_buffers) / (1024 * 1024));
    }
#endif
    return 0;
}
