/*
 * tts_bench.c - End-to-end TTS pipeline benchmark
 *
 * Runs the full TTS pipeline (decode + vocoder) directly in-process,
 * bypassing HTTP, to measure pure synthesis performance across text lengths.
 *
 * Reports: decode time, vocoder time, total time, audio duration, RTF.
 * Uses seed + single-threaded GEMM for deterministic output (same as server).
 *
 * Usage:
 *   tts-bench.exe [--model <tts-model-dir>] [--fp16] [--runs N] [--threads N] [--seed N]
 *
 * Build:
 *   build.bat ttsbench
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "tts_pipeline.h"
#include "tts_vocoder.h"
#include "qwen_asr_kernels.h"
#include "platform.h"
#include "config.h"

/* Required by tts_native.c */
volatile int g_shutdown = 0;

/* ========================================================================
 * Benchmark texts — range of lengths for RTF scaling analysis
 * ======================================================================== */

typedef struct {
    const char *id;
    const char *text;
} bench_case_t;

static const bench_case_t BENCH_CASES[] = {
    {
        "4s",
        "The quick brown fox jumps over the lazy dog near the riverbank."
    },
    {
        "6s",
        "She sells seashells by the seashore, and the shells she sells "
        "are seashells, I'm sure."
    },
    {
        "8s",
        "The old lighthouse keeper climbed the spiral staircase, his weathered "
        "hands gripping the iron railing as wind howled through the cracks "
        "in the ancient stone walls."
    },
    {
        "10s",
        "In the beginning, the universe was created. This has made a lot of "
        "people very angry and been widely regarded as a bad move. Many were "
        "increasingly of the opinion that they had all made a big mistake "
        "in coming down from the trees in the first place."
    },
    {
        "12s",
        "It was a bright cold day in April, and the clocks were striking thirteen. "
        "Winston Smith, his chin nuzzled into his breast in an effort to escape "
        "the vile wind, slipped quickly through the glass doors of Victory Mansions, "
        "though not quickly enough to prevent a swirl of gritty dust from entering "
        "along with him."
    },
    {
        "16s",
        "It was a bright cold day in April, and the clocks were striking thirteen. "
        "Winston Smith, his chin nuzzled into his breast in an effort to escape "
        "the vile wind, slipped quickly through the glass doors of Victory Mansions, "
        "though not quickly enough to prevent a swirl of gritty dust from entering "
        "along with him. The hallway smelt of boiled cabbage and old rag mats. "
        "At one end of it a coloured poster, too large for indoor display, had been "
        "tacked to the wall."
    },
};

#define NUM_CASES (int)(sizeof(BENCH_CASES) / sizeof(BENCH_CASES[0]))

/* ========================================================================
 * Timing data per run
 * ======================================================================== */

typedef struct {
    double total_ms;     /* total synthesis (decode + vocoder + WAV encode) */
    int    n_steps;      /* autoregressive steps */
    int    n_samples;    /* PCM samples output */
    double audio_sec;    /* n_samples / 24000 */
    double rtf;          /* total_ms / (audio_sec * 1000) */
} run_timing_t;

/* ========================================================================
 * Helpers
 * ======================================================================== */

static double median_of(double *vals, int n) {
    /* Insertion sort — n is tiny */
    for (int i = 1; i < n; i++) {
        double v = vals[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > v) {
            vals[j + 1] = vals[j];
            j--;
        }
        vals[j + 1] = v;
    }
    if (n % 2 == 1) return vals[n / 2];
    return (vals[n / 2 - 1] + vals[n / 2]) / 2.0;
}

/* ========================================================================
 * Benchmark one case
 * ======================================================================== */

static int bench_case(TtsPipeline *tts, const bench_case_t *bc,
                       int runs, int seed, int threads,
                       run_timing_t *out_median) {
    run_timing_t *timings = (run_timing_t *)calloc(runs, sizeof(run_timing_t));
    if (!timings) return -1;

    printf("  %-12s \"%.*s%s\"\n", bc->id, 60, bc->text,
           strlen(bc->text) > 60 ? "..." : "");

    for (int r = 0; r < runs; r++) {
        /* Seed RNG and force single-threaded for deterministic GEMM,
         * matching the server's behavior when seed is specified. */
        srand((unsigned int)seed);
        qwen_set_threads(1);

        TtsResult result;
        int rc = tts_pipeline_synthesize(tts, bc->text,
                                          NULL, NULL,    /* voice, language */
                                          0.3f, 50,      /* temperature, top_k */
                                          1.0f,          /* speed */
                                          NULL, NULL,    /* progress callback */
                                          &result);

        /* Restore threads for next run */
        qwen_set_threads(threads);

        if (rc != 0) {
            fprintf(stderr, "    ERROR: synthesis failed on run %d\n", r + 1);
            free(timings);
            return -1;
        }

        timings[r].total_ms   = result.elapsed_ms;
        timings[r].n_steps    = result.n_steps;
        timings[r].n_samples  = result.n_samples;
        timings[r].audio_sec  = (double)result.n_samples / 24000.0;
        timings[r].rtf        = result.elapsed_ms / (timings[r].audio_sec * 1000.0);

        free(result.wav_data);

        printf("    run %d: %6.0f ms | %3d steps | %5.1fs audio | RTF %.3f\n",
               r + 1, timings[r].total_ms,
               timings[r].n_steps, timings[r].audio_sec, timings[r].rtf);
    }

    /* Compute medians */
    double *vals = (double *)malloc(runs * sizeof(double));

    for (int r = 0; r < runs; r++) vals[r] = timings[r].total_ms;
    out_median->total_ms = median_of(vals, runs);

    for (int r = 0; r < runs; r++) vals[r] = timings[r].rtf;
    out_median->rtf = median_of(vals, runs);

    /* All runs should be identical with seed — use first */
    out_median->n_steps   = timings[0].n_steps;
    out_median->n_samples = timings[0].n_samples;
    out_median->audio_sec = timings[0].audio_sec;

    free(vals);
    free(timings);
    return 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */

static void print_usage(const char *prog) {
    printf("TTS Pipeline Benchmark\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --model <dir>    TTS model directory (default: from config.json)\n");
    printf("  --fp16           Use FP16 GPU weights (default: from config.json)\n");
    printf("  --runs <N>       Benchmark iterations per case (default: 3)\n");
    printf("  --warmup <N>     Warmup iterations (default: 1)\n");
    printf("  --threads <N>    CPU threads for vocoder (default: 4)\n");
    printf("  --seed <N>       Random seed for determinism (default: 42)\n");
}

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

    /* Load config.json defaults */
    app_config_t cfg;
    config_load(&cfg, argv[0]);

    const char *model_dir = cfg.tts_model[0] ? cfg.tts_model : NULL;
    /* TTS defaults to FP16 on GPU builds; config tts_fp16 overrides, legacy fp16 as fallback */
#ifdef USE_CUBLAS
    int fp16 = cfg.tts_fp16 == 0 ? 0 : 1;
#else
    int fp16 = 0;
#endif
    int runs = 3;
    int warmup = 1;
    int threads = cfg.threads > 0 ? cfg.threads : 4;
    int seed = 42;

    for (int i = 1; i < argc; i++) {
        const char *val;
        if ((val = parse_arg(argv[i], "--model")) != NULL) {
            model_dir = val;
        } else if (strcmp(argv[i], "--fp16") == 0) {
            fp16 = 1;
        } else if (strcmp(argv[i], "--no-fp16") == 0) {
            fp16 = 0;
        } else if ((val = parse_arg(argv[i], "--runs")) != NULL) {
            runs = atoi(val);
            if (runs < 1) runs = 1;
        } else if ((val = parse_arg(argv[i], "--warmup")) != NULL) {
            warmup = atoi(val);
            if (warmup < 0) warmup = 0;
        } else if ((val = parse_arg(argv[i], "--threads")) != NULL) {
            threads = atoi(val);
            if (threads < 1) threads = 1;
        } else if ((val = parse_arg(argv[i], "--seed")) != NULL) {
            seed = atoi(val);
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
        fprintf(stderr, "Error: --model is required (or set tts_model in config.json)\n\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Initialize */
    printf("=== TTS Pipeline Benchmark ===\n");
    printf("Model:   %s\n", model_dir);
    printf("FP16:    %s\n", fp16 ? "yes" : "no");
    printf("Threads: %d\n", threads);
    printf("Runs:    %d (+ %d warmup)\n", runs, warmup);
    printf("Seed:    %d\n\n", seed);

    qwen_set_threads(threads);

    printf("Loading model...\n");
    double t_load = platform_time_ms();

    TtsPipeline tts;
    if (tts_pipeline_init(&tts, model_dir, fp16, 0) != 0) {
        fprintf(stderr, "Error: failed to initialize TTS pipeline from %s\n", model_dir);
        return 1;
    }

    printf("Model loaded in %.0f ms\n\n", platform_time_ms() - t_load);

    /* Warmup */
    if (warmup > 0) {
        printf("Warmup (%d run%s)...\n", warmup, warmup > 1 ? "s" : "");
        for (int w = 0; w < warmup; w++) {
            srand((unsigned int)seed);
            qwen_set_threads(1);
            TtsResult result;
            int rc = tts_pipeline_synthesize(&tts, BENCH_CASES[0].text,
                                              NULL, NULL, 0.3f, 50, 1.0f,
                                              NULL, NULL, &result);
            qwen_set_threads(threads);
            if (rc == 0) {
                printf("  warmup %d: %d steps, %.1fs audio, %.0f ms\n",
                       w + 1, result.n_steps,
                       (double)result.n_samples / 24000.0, result.elapsed_ms);
                free(result.wav_data);
            }
        }
        printf("\n");
    }

    /* Run benchmarks */
    printf("Benchmarking %d cases, %d runs each:\n\n", NUM_CASES, runs);

    run_timing_t medians[NUM_CASES];
    memset(medians, 0, sizeof(medians));
    int ok_count = 0;

    for (int c = 0; c < NUM_CASES; c++) {
        if (bench_case(&tts, &BENCH_CASES[c], runs, seed, threads, &medians[c]) == 0) {
            ok_count++;
        }
        printf("\n");
    }

    /* Summary table */
    printf("=== Summary (median of %d runs, seed=%d) ===\n\n", runs, seed);
    printf("%-12s %5s %6s %8s %7s %5s\n",
           "Case", "Steps", "Audio", "Total", "RTF", "chars");
    printf("%-12s %5s %6s %8s %7s %5s\n",
           "------------", "-----", "------", "--------",
           "-------", "-----");

    for (int c = 0; c < NUM_CASES; c++) {
        if (medians[c].n_steps == 0) continue;

        run_timing_t *m = &medians[c];
        int text_len = (int)strlen(BENCH_CASES[c].text);

        printf("%-12s %5d %5.1fs %6.0f ms %5.3fx %5d\n",
               BENCH_CASES[c].id,
               m->n_steps, m->audio_sec,
               m->total_ms,
               m->rtf, text_len);
    }

    /* Per-step cost analysis */
    if (ok_count >= 2) {
        printf("\n=== Per-step cost ===\n\n");
        printf("%-12s %5s %6s %10s\n",
               "Case", "Steps", "Audio", "ms/step");
        printf("%-12s %5s %6s %10s\n",
               "------------", "-----", "------", "----------");

        for (int c = 0; c < NUM_CASES; c++) {
            if (medians[c].n_steps == 0) continue;
            run_timing_t *m = &medians[c];
            double ms_per_step = m->total_ms / m->n_steps;
            printf("%-12s %5d %5.1fs %8.1f ms\n",
                   BENCH_CASES[c].id,
                   m->n_steps, m->audio_sec, ms_per_step);
        }
    }

    printf("\n");
    tts_pipeline_free(&tts);
    printf("Done.\n");
    return 0;
}
