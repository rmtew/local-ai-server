/*
 * local-ai-server -- OpenAI-compatible local inference server
 *
 * Wraps qwen-asr behind a standard /v1/audio/transcriptions endpoint.
 * Single-threaded, personal use.
 *
 * Usage: local-ai-server.exe --model=<dir> [--port=N] [--language=<lang>]
 *                             [--threads=N] [--verbose] [--help]
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

/* Global server for Ctrl+C handler */
static HttpServer g_server;

static BOOL WINAPI ctrl_handler(DWORD type) {
    (void)type;
    printf("\nShutting down...\n");
    http_server_shutdown(&g_server);
    return TRUE;
}

static void print_usage(const char *prog) {
    printf("Usage: %s --model=<dir> [options]\n\n", prog);
    printf("OpenAI-compatible local inference server.\n\n");
    printf("Options:\n");
    printf("  --model=<dir>      Path to qwen-asr model directory (required)\n");
    printf("  --port=<N>         Listen port (default: 8090)\n");
    printf("  --language=<lang>  Force language (default: auto-detect)\n");
    printf("  --threads=<N>      CPU threads for inference (default: 4)\n");
    printf("  --verbose          Enable verbose logging\n");
    printf("  --help             Show this help message\n");
    printf("\nSupported languages: %s\n", qwen_supported_languages_csv());
    printf("\nEndpoints:\n");
    printf("  POST /v1/audio/transcriptions  Transcribe audio (OpenAI-compatible)\n");
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

    const char *model_dir = NULL;
    int port = 8090;
    const char *language = NULL;
    int threads = 4;
    int verbose = 0;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        const char *val;
        if ((val = parse_arg(argv[i], "--model")) != NULL) {
            model_dir = val;
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

    if (!model_dir) {
        fprintf(stderr, "Error: --model=<dir> is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Set thread count (must be before qwen_load) */
    qwen_set_threads(threads);

    if (verbose) {
        qwen_verbose = 1;
    }

    /* Load model */
    printf("Loading model from: %s\n", model_dir);
    qwen_ctx_t *asr_ctx = qwen_load(model_dir);
    if (!asr_ctx) {
        fprintf(stderr, "Error: failed to load model from %s\n", model_dir);
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

    /* Initialize HTTP server */
    memset(&g_server, 0, sizeof(g_server));
    if (http_server_init(&g_server, port) != 0) {
        qwen_free(asr_ctx);
        return 1;
    }

    /* Register Ctrl+C handler */
    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    /* Print banner */
    printf("\n");
    printf("=== local-ai-server ===\n");
    printf("Port:     %d\n", port);
    printf("Model:    %s\n", model_dir);
    printf("Language: %s\n", language ? language : "(auto-detect)");
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
    handler_ctx.asr_ctx = asr_ctx;
    handler_ctx.verbose = verbose;

    http_server_run(&g_server, asr_handle_request, &handler_ctx);

    /* Cleanup */
    printf("Cleaning up...\n");
    qwen_free(asr_ctx);
    return 0;
}
