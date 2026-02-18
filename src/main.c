/*
 * local-ai-server -- OpenAI-compatible local inference server
 *
 * Serves ASR (Qwen-ASR) and TTS (Qwen3-TTS) behind OpenAI-compatible endpoints.
 * Single-threaded, personal use.
 *
 * Usage: local-ai-server.exe [--model=<dir>] [--tts-model=<dir>] [--port=N]
 *                             [--language=<lang>] [--threads=N] [--no-fp16]
 *                             [--no-fp16-asr] [--int8-asr] [--int8-tts]
 *                             [--verbose] [--help]
 */

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <signal.h>
#endif

#include "http.h"
#include "handler_asr.h"
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_gpu.h"
#include "tts_pipeline.h"
#include "config.h"
#include "platform.h"

/* Global server for Ctrl+C handler */
static HttpServer g_server;

/* Shutdown flag checked by long-running operations (e.g. TTS decode loop) */
volatile int g_shutdown = 0;

#ifdef _WIN32
static BOOL WINAPI ctrl_handler(DWORD type) {
    (void)type;
    printf("\nShutting down...\n");
    g_shutdown = 1;
    http_server_shutdown(&g_server);
    return TRUE;
}
#else
static void signal_handler(int sig) {
    (void)sig;
    printf("\nShutting down...\n");
    g_shutdown = 1;
    http_server_shutdown(&g_server);
}
#endif

static void print_usage(const char *prog) {
    printf("Usage: %s [--model=<dir>] [--tts-model=<dir>] [options]\n\n", prog);
    printf("OpenAI-compatible local inference server.\n");
    printf("At least one of --model or --tts-model must be specified.\n\n");
    printf("Options:\n");
    printf("  --model=<dir>      Path to qwen-asr model directory (ASR)\n");
    printf("  --tts-model=<dir>  Path to qwen3-tts model directory (TTS)\n");
    printf("  --port=<N>         Listen port (default: 8090)\n");
    printf("  --language=<lang>  Force ASR language (default: auto-detect)\n");
    printf("  --threads=<N>      CPU threads for inference (default: 4)\n");
    printf("  --no-fp16          Disable TTS FP16 (TTS uses FP16 by default on GPU)\n");
    printf("  --no-fp16-asr      Disable ASR FP16 (ASR uses FP16 by default on GPU)\n");
    printf("  --int8-asr         Use INT8 quantization for ASR decoder weights\n");
    printf("  --int8-tts         Use INT8 quantization for TTS talker weights\n");
    printf("  --tts-max-steps=<N> Max TTS decode steps (default 200, ~16s audio)\n");
    printf("  --verbose          Enable verbose logging\n");
    printf("  --help             Show this help message\n");
    printf("\nASR languages: %s\n", qwen_supported_languages_csv());
    printf("\nEndpoints:\n");
    printf("  POST /v1/audio/transcriptions  Transcribe audio (requires --model)\n");
    printf("  POST /v1/audio/speech           Synthesize speech (requires --tts-model)\n");
    printf("  GET  /v1/models                List available models\n");
    printf("  GET  /health                   Health check\n");
}

/* Parse --key=value style arguments */
static const char *parse_arg(const char *arg, const char *key) {
    size_t klen = strlen(key);
    if (strncmp(arg, key, klen) == 0 && arg[klen] == '=') {
        return arg + klen + 1;
    }
    return NULL;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    /* Load config.json defaults (optional) */
    app_config_t cfg;
    config_load(&cfg, argv[0]);

    const char *model_dir = cfg.model[0] ? cfg.model : NULL;
    const char *tts_model_dir = cfg.tts_model[0] ? cfg.tts_model : NULL;
    int port = cfg.port > 0 ? cfg.port : 8090;
    const char *language = cfg.language[0] ? cfg.language : NULL;
    int threads = cfg.threads > 0 ? cfg.threads : 4;
    int verbose = cfg.verbose == 1 ? 1 : 0;
#ifdef USE_CUBLAS
    int tts_fp16 = cfg.tts_fp16 == 0 ? 0 : 1;  /* default ON for GPU builds */
    int asr_fp16 = cfg.asr_fp16 == 0 ? 0 : 1;   /* default ON for GPU builds */
    int asr_int8 = cfg.asr_int8 == 1 ? 1 : 0;   /* default OFF */
    int tts_int8 = cfg.tts_int8 == 1 ? 1 : 0;   /* default OFF */
#else
    int tts_fp16 = 0;
    int asr_fp16 = 0;
    int asr_int8 = 0;
    int tts_int8 = 0;
#endif
    int tts_max_steps = cfg.tts_max_steps;  /* 0 = use default (TTS_MAX_DECODE_STEPS) */

    /* Parse arguments (override config) */
    for (int i = 1; i < argc; i++) {
        const char *val;
        if ((val = parse_arg(argv[i], "--model")) != NULL) {
            model_dir = val;
        } else if ((val = parse_arg(argv[i], "--tts-model")) != NULL) {
            tts_model_dir = val;
        } else if ((val = parse_arg(argv[i], "--port")) != NULL) {
            port = atoi(val);
        } else if ((val = parse_arg(argv[i], "--language")) != NULL) {
            language = val;
        } else if ((val = parse_arg(argv[i], "--threads")) != NULL) {
            threads = atoi(val);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--no-fp16") == 0) {
            tts_fp16 = 0;
        } else if (strcmp(argv[i], "--no-fp16-asr") == 0) {
            asr_fp16 = 0;
        } else if (strcmp(argv[i], "--fp16-asr") == 0) {
            asr_fp16 = 1;  /* legacy: still accepted */
        } else if (strcmp(argv[i], "--int8-asr") == 0) {
            asr_int8 = 1;
        } else if (strcmp(argv[i], "--int8-tts") == 0) {
            tts_int8 = 1;
        } else if ((val = parse_arg(argv[i], "--tts-max-steps")) != NULL) {
            tts_max_steps = atoi(val);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir && !tts_model_dir) {
        fprintf(stderr, "Error: at least one of --model or --tts-model is required\n");
        fprintf(stderr, "Specify via CLI args or config.json\n\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Set thread count (must be before qwen_load) */
    qwen_set_threads(threads);

    if (verbose) {
        qwen_verbose = 1;
    }

    /* Per-model memory tracking */
#ifdef USE_CUBLAS
    qwen_gpu_stats_t gpu_stats_asr = {0};
    qwen_gpu_stats_t gpu_stats_tts = {0};
    qwen_gpu_stats_t gpu_stats_total = {0};
    extern qwen_gpu_ctx_t *g_gpu_ctx;
#endif
    size_t rss_baseline = platform_rss_bytes();
    size_t rss_after_asr = 0;
    size_t rss_after_tts = 0;

    /* Load ASR model (optional) */
    qwen_ctx_t *asr_ctx = NULL;
    if (model_dir) {
        printf("Loading ASR model from: %s\n", model_dir);
#ifdef USE_CUBLAS
        if (asr_int8) {
            qwen_set_gpu_int8(1);
            qwen_set_gpu_fp16(0);  /* INT8 takes priority */
        } else {
            qwen_set_gpu_fp16(asr_fp16);
        }
#endif
        asr_ctx = qwen_load(model_dir);
        if (!asr_ctx) {
            fprintf(stderr, "Error: failed to load ASR model from %s\n", model_dir);
            return 1;
        }
        rss_after_asr = platform_rss_bytes();
#ifdef USE_CUBLAS
        qwen_gpu_get_stats(g_gpu_ctx, &gpu_stats_asr);
#endif

        /* Set forced language if specified */
        if (language) {
            if (qwen_set_force_language(asr_ctx, language) != 0) {
                fprintf(stderr, "Error: unsupported language '%s'\n", language);
                fprintf(stderr, "Supported: %s\n", qwen_supported_languages_csv());
                qwen_free(asr_ctx);
                return 1;
            }
        }
    }

    /* Load TTS model (optional) */
    TtsPipeline *tts_pipeline = NULL;
    if (tts_int8) tts_fp16 = 0;  /* INT8 takes priority over FP16 */
    if (tts_model_dir) {
        tts_pipeline = (TtsPipeline *)calloc(1, sizeof(TtsPipeline));
        if (!tts_pipeline) {
            fprintf(stderr, "Error: failed to allocate TTS pipeline\n");
            if (asr_ctx) qwen_free(asr_ctx);
            return 1;
        }
        if (tts_pipeline_init(tts_pipeline, tts_model_dir, tts_fp16, tts_int8, verbose) != 0) {
            fprintf(stderr, "Error: failed to load TTS model from %s\n", tts_model_dir);
            free(tts_pipeline);
            if (asr_ctx) qwen_free(asr_ctx);
            return 1;
        }
        if (tts_max_steps > 0) {
            tts_pipeline->native->max_steps = tts_max_steps;
            if (verbose)
                printf("TTS max decode steps: %d (~%.1fs audio)\n",
                       tts_max_steps, tts_max_steps * 0.08);
        }
        rss_after_tts = platform_rss_bytes();
#ifdef USE_CUBLAS
        qwen_gpu_get_stats(g_gpu_ctx, &gpu_stats_total);
        /* TTS delta = total - ASR */
        gpu_stats_tts.n_weights = gpu_stats_total.n_weights - gpu_stats_asr.n_weights;
        gpu_stats_tts.n_weights_f32 = gpu_stats_total.n_weights_f32 - gpu_stats_asr.n_weights_f32;
        gpu_stats_tts.n_weights_f16 = gpu_stats_total.n_weights_f16 - gpu_stats_asr.n_weights_f16;
        gpu_stats_tts.n_weights_int8 = gpu_stats_total.n_weights_int8 - gpu_stats_asr.n_weights_int8;
        gpu_stats_tts.vram_weights = gpu_stats_total.vram_weights - gpu_stats_asr.vram_weights;
#endif
    }

    /* Initialize HTTP server */
    memset(&g_server, 0, sizeof(g_server));
    if (http_server_init(&g_server, port) != 0) {
        if (asr_ctx) qwen_free(asr_ctx);
        if (tts_pipeline) { tts_pipeline_free(tts_pipeline); free(tts_pipeline); }
        return 1;
    }

    /* Register Ctrl+C handler */
#ifdef _WIN32
    SetConsoleCtrlHandler(ctrl_handler, TRUE);
#else
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
#endif

    /* Print banner */
    printf("\n");
    printf("=== local-ai-server ===\n");
    if (cfg.loaded)
        printf("Config:   %s\n", cfg.config_path);
    printf("Port:     %d\n", port);
    if (model_dir) {
        printf("ASR:      %s\n", model_dir);
        {
            int ram_mb = rss_after_asr > rss_baseline
                       ? (int)((rss_after_asr - rss_baseline) / (1024 * 1024)) : 0;
#ifdef USE_CUBLAS
            if (gpu_stats_asr.n_weights > 0) {
                if (gpu_stats_asr.n_weights_int8 > 0) {
                    int vram_mb = (int)(gpu_stats_asr.vram_weights / (1024 * 1024));
                    if (gpu_stats_asr.n_weights_f16 > 0)
                        printf("          %d MB RAM, %d MB VRAM (%d weights: %d F32, %d FP16, %d INT8)\n",
                               ram_mb, vram_mb, gpu_stats_asr.n_weights,
                               gpu_stats_asr.n_weights_f32, gpu_stats_asr.n_weights_f16,
                               gpu_stats_asr.n_weights_int8);
                    else
                        printf("          %d MB RAM, %d MB VRAM (%d weights: %d F32, %d INT8)\n",
                               ram_mb, vram_mb, gpu_stats_asr.n_weights,
                               gpu_stats_asr.n_weights_f32, gpu_stats_asr.n_weights_int8);
                } else if (gpu_stats_asr.n_weights_f16 > 0)
                    printf("          %d MB RAM, %d MB VRAM (%d weights: %d F32, %d FP16)\n",
                           ram_mb,
                           (int)(gpu_stats_asr.vram_weights / (1024 * 1024)),
                           gpu_stats_asr.n_weights,
                           gpu_stats_asr.n_weights_f32, gpu_stats_asr.n_weights_f16);
                else
                    printf("          %d MB RAM, %d MB VRAM (%d weights, F32)\n",
                           ram_mb,
                           (int)(gpu_stats_asr.vram_weights / (1024 * 1024)),
                           gpu_stats_asr.n_weights);
            } else
#endif
                printf("          %d MB RAM\n", ram_mb);
        }
    }
    if (tts_model_dir) {
        printf("TTS:      %s\n", tts_model_dir);
        {
            size_t rss_before_tts = rss_after_asr > 0 ? rss_after_asr : rss_baseline;
            int ram_mb = rss_after_tts > rss_before_tts
                       ? (int)((rss_after_tts - rss_before_tts) / (1024 * 1024)) : 0;
#ifdef USE_CUBLAS
            if (gpu_stats_tts.n_weights > 0) {
                int vram_mb = (int)(gpu_stats_tts.vram_weights / (1024 * 1024));
                if (gpu_stats_tts.n_weights_int8 > 0) {
                    if (gpu_stats_tts.n_weights_f16 > 0)
                        printf("          %d MB RAM, %d MB VRAM (%d weights: %d F32, %d FP16, %d INT8)\n",
                               ram_mb, vram_mb, gpu_stats_tts.n_weights,
                               gpu_stats_tts.n_weights_f32, gpu_stats_tts.n_weights_f16,
                               gpu_stats_tts.n_weights_int8);
                    else
                        printf("          %d MB RAM, %d MB VRAM (%d weights: %d F32, %d INT8)\n",
                               ram_mb, vram_mb, gpu_stats_tts.n_weights,
                               gpu_stats_tts.n_weights_f32, gpu_stats_tts.n_weights_int8);
                } else if (gpu_stats_tts.n_weights_f16 > 0)
                    printf("          %d MB RAM, %d MB VRAM (%d weights: %d F32, %d FP16)\n",
                           ram_mb, vram_mb, gpu_stats_tts.n_weights,
                           gpu_stats_tts.n_weights_f32, gpu_stats_tts.n_weights_f16);
                else
                    printf("          %d MB RAM, %d MB VRAM (%d weights, F32)\n",
                           ram_mb, vram_mb, gpu_stats_tts.n_weights);
            } else
#endif
                printf("          %d MB RAM\n", ram_mb);
        }
    }
    if (language)
        printf("Language: %s\n", language);
    printf("Threads:  %d\n", threads);
    {
        size_t rss_final = rss_after_tts > 0 ? rss_after_tts
                         : rss_after_asr > 0 ? rss_after_asr : 0;
        if (rss_final > rss_baseline)
            printf("RAM:      %d MB total\n",
                   (int)((rss_final - rss_baseline) / (1024 * 1024)));
    }
#ifdef USE_CUBLAS
    {
        /* Use total if both models loaded, otherwise whichever is available */
        qwen_gpu_stats_t *gs = gpu_stats_total.n_weights > 0 ? &gpu_stats_total
                             : gpu_stats_asr.n_weights > 0 ? &gpu_stats_asr
                             : &gpu_stats_tts;
        if (gs->n_weights > 0) {
            printf("GPU:      cuBLAS — %d weights, %d MB VRAM\n",
                   gs->n_weights,
                   (int)(gs->vram_weights / (1024 * 1024)));
        } else {
            printf("GPU:      cuBLAS enabled\n");
        }
    }
#ifdef USE_CUDA_KERNELS
    printf("          + custom CUDA kernels\n");
#endif
#else
    printf("GPU:      disabled\n");
#endif
    printf("\nListening on http://localhost:%d\n", port);
    printf("Press Ctrl+C to stop.\n\n");

    /* Run server */
    HandlerContext handler_ctx;
    memset(&handler_ctx, 0, sizeof(handler_ctx));
    handler_ctx.asr_ctx = asr_ctx;
    handler_ctx.verbose = verbose;
    handler_ctx.threads = threads;
    handler_ctx.tts = tts_pipeline;

    http_server_run(&g_server, asr_handle_request, &handler_ctx);

    /* Cleanup */
    printf("Cleaning up...\n");
    if (asr_ctx) qwen_free(asr_ctx);
    if (tts_pipeline) {
        tts_pipeline_free(tts_pipeline);
        free(tts_pipeline);
    }
    return 0;
}
