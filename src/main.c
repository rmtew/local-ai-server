/*
 * local-ai-server -- OpenAI-compatible local inference server
 *
 * Serves ASR (Qwen-ASR) and TTS (Qwen3-TTS) behind OpenAI-compatible endpoints.
 * Single-threaded, personal use.
 *
 * Usage: local-ai-server.exe [--model=<dir>] [--tts-model=<dir>] [--port=N]
 *                             [--language=<lang>] [--threads=N] [--verbose] [--help]
 */

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "http.h"
#include "handler_asr.h"
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"

#ifdef USE_ORT
#include "tts_pipeline.h"
#endif

/* Global server for Ctrl+C handler */
static HttpServer g_server;

/* Shutdown flag checked by long-running operations (e.g. TTS decode loop) */
volatile int g_shutdown = 0;

static BOOL WINAPI ctrl_handler(DWORD type) {
    (void)type;
    printf("\nShutting down...\n");
    g_shutdown = 1;
    http_server_shutdown(&g_server);
    return TRUE;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [--model=<dir>] [--tts-model=<dir>] [options]\n\n", prog);
    printf("OpenAI-compatible local inference server.\n");
    printf("At least one of --model or --tts-model must be specified.\n\n");
    printf("Options:\n");
    printf("  --model=<dir>      Path to qwen-asr model directory (ASR)\n");
    printf("  --tts-model=<dir>  Path to qwen3-tts ONNX model directory (TTS)\n");
    printf("  --port=<N>         Listen port (default: 8090)\n");
    printf("  --language=<lang>  Force ASR language (default: auto-detect)\n");
    printf("  --threads=<N>      CPU threads for inference (default: 4)\n");
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
    SetConsoleOutputCP(CP_UTF8);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const char *model_dir = NULL;
    const char *tts_model_dir = NULL;
    int port = 8090;
    const char *language = NULL;
    int threads = 4;
    int verbose = 0;

    /* Parse arguments */
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
        fprintf(stderr, "Error: at least one of --model or --tts-model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Set thread count (must be before qwen_load) */
    qwen_set_threads(threads);

    if (verbose) {
        qwen_verbose = 1;
    }

    /* Load ASR model (optional) */
    qwen_ctx_t *asr_ctx = NULL;
    if (model_dir) {
        printf("Loading ASR model from: %s\n", model_dir);
        asr_ctx = qwen_load(model_dir);
        if (!asr_ctx) {
            fprintf(stderr, "Error: failed to load ASR model from %s\n", model_dir);
            return 1;
        }

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
#ifdef USE_ORT
    TtsPipeline *tts_pipeline = NULL;
    if (tts_model_dir) {
        tts_pipeline = (TtsPipeline *)calloc(1, sizeof(TtsPipeline));
        if (!tts_pipeline) {
            fprintf(stderr, "Error: failed to allocate TTS pipeline\n");
            if (asr_ctx) qwen_free(asr_ctx);
            return 1;
        }
        if (tts_pipeline_init(tts_pipeline, tts_model_dir, verbose) != 0) {
            fprintf(stderr, "Error: failed to load TTS model from %s\n", tts_model_dir);
            free(tts_pipeline);
            if (asr_ctx) qwen_free(asr_ctx);
            return 1;
        }
    }
#else
    if (tts_model_dir) {
        fprintf(stderr, "Warning: --tts-model ignored (built without ONNX Runtime support)\n");
    }
#endif

    /* Initialize HTTP server */
    memset(&g_server, 0, sizeof(g_server));
    if (http_server_init(&g_server, port) != 0) {
        if (asr_ctx) qwen_free(asr_ctx);
#ifdef USE_ORT
        if (tts_pipeline) { tts_pipeline_free(tts_pipeline); free(tts_pipeline); }
#endif
        return 1;
    }

    /* Register Ctrl+C handler */
    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    /* Print banner */
    printf("\n");
    printf("=== local-ai-server ===\n");
    printf("Port:     %d\n", port);
    if (model_dir)
        printf("ASR:      %s\n", model_dir);
    if (tts_model_dir)
        printf("TTS:      %s\n", tts_model_dir);
    if (language)
        printf("Language: %s\n", language);
    printf("Threads:  %d\n", threads);
#ifdef USE_CUBLAS
    printf("GPU:      cuBLAS enabled\n");
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
#ifdef USE_ORT
    handler_ctx.tts = tts_pipeline;
#endif

    http_server_run(&g_server, asr_handle_request, &handler_ctx);

    /* Cleanup */
    printf("Cleaning up...\n");
    if (asr_ctx) qwen_free(asr_ctx);
#ifdef USE_ORT
    if (tts_pipeline) {
        tts_pipeline_free(tts_pipeline);
        free(tts_pipeline);
    }
#endif
    return 0;
}
