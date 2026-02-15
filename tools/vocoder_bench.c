/*
 * vocoder_bench.c - Standalone vocoder benchmark and verification tool
 *
 * Loads vocoder weights, runs the vocoder on saved codec tokens, reports
 * per-stage timing, and compares output against a reference file.
 *
 * Usage:
 *   vocoder-bench.exe --model <tokenizer-dir> --codes <file.codes>
 *                     [--ref <ref.raw>] [--runs N] [--out <output.raw>]
 *
 *   vocoder-bench.exe --model <tts-model-dir> --generate
 *                     [--out-codes <file.codes>] [--out <file.raw>]
 *
 * File formats:
 *   .codes: raw int64 array, [n_steps * 16] (row-major, 16 codebooks)
 *   .raw:   raw float32 array, [n_samples] (24 kHz mono)
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "tts_vocoder.h"
#include "qwen_asr_kernels.h"

#ifdef USE_ORT
#include "tts_pipeline.h"
#endif

/* Required by tts_native.c */
volatile int g_shutdown = 0;

/* ========================================================================
 * Comparison metrics (built-in, no Python dependency)
 * ======================================================================== */

typedef struct {
    double correlation;
    double snr_db;
    double max_diff;
    int n_samples;
    int ref_samples;
} compare_result_t;

static void compare_audio(const float *test, int test_n,
                           const float *ref, int ref_n,
                           compare_result_t *result) {
    memset(result, 0, sizeof(*result));
    result->n_samples = test_n;
    result->ref_samples = ref_n;

    int n = (test_n < ref_n) ? test_n : ref_n;
    if (n == 0) return;

    /* Pearson correlation */
    double sum_t = 0, sum_r = 0;
    for (int i = 0; i < n; i++) {
        sum_t += test[i];
        sum_r += ref[i];
    }
    double mean_t = sum_t / n;
    double mean_r = sum_r / n;

    double cov = 0, var_t = 0, var_r = 0;
    double noise_pow = 0, sig_pow = 0;
    double max_d = 0;
    for (int i = 0; i < n; i++) {
        double dt = test[i] - mean_t;
        double dr = ref[i] - mean_r;
        cov += dt * dr;
        var_t += dt * dt;
        var_r += dr * dr;

        double diff = fabs((double)test[i] - (double)ref[i]);
        if (diff > max_d) max_d = diff;

        noise_pow += (test[i] - ref[i]) * (double)(test[i] - ref[i]);
        sig_pow += ref[i] * (double)ref[i];
    }

    result->max_diff = max_d;

    if (var_t > 0 && var_r > 0) {
        result->correlation = cov / (sqrt(var_t) * sqrt(var_r));
    }

    if (noise_pow > 0 && sig_pow > 0) {
        result->snr_db = 10.0 * log10(sig_pow / noise_pow);
    } else if (noise_pow == 0 && sig_pow > 0) {
        result->snr_db = INFINITY;
    }
}

/* ========================================================================
 * File I/O
 * ======================================================================== */

static int64_t *load_codes(const char *path, int *out_n_steps) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open codes file '%s'\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (sz % (int64_t)(16 * sizeof(int64_t)) != 0) {
        fprintf(stderr, "Error: codes file size %ld not a multiple of %d (16 codebooks * 8 bytes)\n",
                sz, (int)(16 * sizeof(int64_t)));
        fclose(f);
        return NULL;
    }

    int n_steps = (int)(sz / (16 * sizeof(int64_t)));
    int64_t *codes = (int64_t *)malloc(sz);
    if (!codes) {
        fclose(f);
        return NULL;
    }
    fread(codes, 1, sz, f);
    fclose(f);

    *out_n_steps = n_steps;
    return codes;
}

static float *load_raw_audio(const char *path, int *out_n_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open reference file '%s'\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    int n = (int)(sz / sizeof(float));
    float *data = (float *)malloc(sz);
    if (!data) {
        fclose(f);
        return NULL;
    }
    fread(data, 1, sz, f);
    fclose(f);

    *out_n_samples = n;
    return data;
}

static int save_raw(const char *path, const void *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error: cannot write to '%s'\n", path);
        return -1;
    }
    fwrite(data, 1, size, f);
    fclose(f);
    return 0;
}

/* ========================================================================
 * Timing report
 * ======================================================================== */

static void print_timing(const voc_timing_t *t) {
    double total = t->total_ms;
    if (total <= 0) total = 1;

    printf("  RVQ decode:      %7.0f ms (%5.1f%%)\n", t->rvq_ms, t->rvq_ms / total * 100);
    printf("  Pre-conv:        %7.0f ms (%5.1f%%)\n", t->preconv_ms, t->preconv_ms / total * 100);
    printf("  Pre-transformer: %7.0f ms (%5.1f%%)\n", t->xfmr_ms, t->xfmr_ms / total * 100);
    for (int s = 0; s < VOC_UPSAMPLE_STAGES; s++) {
        printf("  Upsample %d:      %7.0f ms (%5.1f%%)\n", s, t->upsample_ms[s], t->upsample_ms[s] / total * 100);
    }
    printf("  BigVGAN init:    %7.0f ms (%5.1f%%)\n", t->bigvgan_init_ms, t->bigvgan_init_ms / total * 100);
    for (int b = 0; b < VOC_BIGVGAN_NUM_BLOCKS; b++) {
        printf("  BigVGAN block %d: %7.0f ms (%5.1f%%)\n", b, t->bigvgan_block_ms[b], t->bigvgan_block_ms[b] / total * 100);
        printf("    tconv:         %7.0f ms (%5.1f%%)\n", t->bigvgan_tconv_ms[b], t->bigvgan_tconv_ms[b] / total * 100);
        printf("    resunits:      %7.0f ms (%5.1f%%)\n", t->bigvgan_res_ms[b], t->bigvgan_res_ms[b] / total * 100);
    }
    printf("  Final:           %7.0f ms (%5.1f%%)\n", t->final_ms, t->final_ms / total * 100);
    printf("  Total:           %7.0f ms\n", t->total_ms);
}

/* Find median of N doubles (modifies array in-place via simple sort). */
static double median_of(double *vals, int n) {
    /* Insertion sort -- n is tiny (1-10) */
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
 * Generate mode: run full TTS pipeline to produce reference codes + audio
 * ======================================================================== */

#ifdef USE_ORT
static int do_generate(const char *model_dir, const char *out_codes_path,
                        const char *out_raw_path) {
    printf("=== Generate mode ===\n");
    printf("Model directory: %s\n", model_dir);

    /* Set deterministic seed before any TTS operations */
    srand(42);

    qwen_set_threads(4);

    TtsPipeline tts;
    if (tts_pipeline_init(&tts, model_dir, 1) != 0) {
        fprintf(stderr, "Error: failed to initialize TTS pipeline from %s\n", model_dir);
        return -1;
    }

    /* Fixed sentence for reproducible benchmarking */
    const char *text = "The quick brown fox jumps over the lazy dog near the riverbank.";
    printf("Synthesizing: \"%s\"\n", text);

    /* Use tts_native_decode to get codec tokens (srand(42) makes sampling deterministic) */
    int n_steps = 0;
    int64_t *codes = NULL;
    int rc = tts_native_decode(tts.native, text, 0.3f, 50, &codes, &n_steps);
    if (rc != 0 || !codes || n_steps == 0) {
        fprintf(stderr, "Error: native decode failed\n");
        tts_pipeline_free(&tts);
        return -1;
    }

    printf("Decode: %d steps (%.1f seconds of audio)\n", n_steps, n_steps * 0.08);

    /* Save codes */
    const char *codes_path = out_codes_path ? out_codes_path : "bench_ref.codes";
    size_t codes_size = (size_t)n_steps * 16 * sizeof(int64_t);
    if (save_raw(codes_path, codes, codes_size) != 0) {
        free(codes);
        tts_pipeline_free(&tts);
        return -1;
    }
    printf("Saved codes: %s (%d steps, %zu bytes)\n", codes_path, n_steps, codes_size);

    /* Run vocoder to get reference audio */
    int n_samples = 0;
    float *audio = tts_vocoder_run(tts.vocoder, codes, n_steps, &n_samples, NULL);
    free(codes);

    if (!audio || n_samples == 0) {
        fprintf(stderr, "Error: vocoder produced no audio\n");
        tts_pipeline_free(&tts);
        return -1;
    }

    /* Save reference audio */
    const char *raw_path = out_raw_path ? out_raw_path : "bench_ref.raw";
    size_t audio_size = (size_t)n_samples * sizeof(float);
    if (save_raw(raw_path, audio, audio_size) != 0) {
        free(audio);
        tts_pipeline_free(&tts);
        return -1;
    }
    printf("Saved audio: %s (%d samples, %.1f seconds, %zu bytes)\n",
           raw_path, n_samples, (double)n_samples / VOC_SAMPLE_RATE, audio_size);

    free(audio);
    tts_pipeline_free(&tts);
    printf("Generate complete.\n");
    return 0;
}
#endif /* USE_ORT */

/* ========================================================================
 * Benchmark mode: vocoder-only timing and comparison
 * ======================================================================== */

static int do_benchmark(const char *model_dir, const char *codes_path,
                         const char *ref_path, const char *out_path, int runs) {
    printf("=== Vocoder Benchmark ===\n");

    /* Auto-detect: if model_dir is the TTS model dir (not the vocoder dir),
     * resolve to the sibling Qwen3-TTS-Tokenizer-12Hz directory.
     * The pipeline uses "../Qwen3-TTS-Tokenizer-12Hz" relative to model_dir. */
    char resolved_dir[512];
    {
        char sibling[512];
        snprintf(sibling, sizeof(sibling), "%s/../Qwen3-TTS-Tokenizer-12Hz/model.safetensors", model_dir);
        FILE *f = fopen(sibling, "rb");
        if (f) {
            fclose(f);
            snprintf(resolved_dir, sizeof(resolved_dir), "%s/../Qwen3-TTS-Tokenizer-12Hz", model_dir);
            printf("Auto-resolved vocoder dir: %s\n", resolved_dir);
        } else {
            /* Assume model_dir already points to Tokenizer-12Hz */
            snprintf(resolved_dir, sizeof(resolved_dir), "%s", model_dir);
        }
    }

    /* Load vocoder */
    tts_vocoder_ctx_t voc;
    if (tts_vocoder_init(&voc, resolved_dir, 1) != 0) {
        fprintf(stderr, "Error: failed to load vocoder from %s\n", resolved_dir);
        return -1;
    }

    /* Load codes */
    int n_steps = 0;
    int64_t *codes = load_codes(codes_path, &n_steps);
    if (!codes) {
        tts_vocoder_free(&voc);
        return -1;
    }

    printf("Codes: %d steps (%.1f seconds of audio)\n",
           n_steps, (double)n_steps * VOC_UPSAMPLE_TOTAL / VOC_SAMPLE_RATE);

    /* Load reference if provided */
    float *ref_audio = NULL;
    int ref_n = 0;
    if (ref_path) {
        ref_audio = load_raw_audio(ref_path, &ref_n);
        if (!ref_audio) {
            fprintf(stderr, "Warning: could not load reference, skipping comparison\n");
        }
    }

    /* Allocate timing arrays for all runs */
    voc_timing_t *timings = (voc_timing_t *)calloc(runs, sizeof(voc_timing_t));
    float *last_audio = NULL;
    int last_n = 0;

    printf("Running %d iteration%s...\n", runs, runs > 1 ? "s" : "");

    for (int r = 0; r < runs; r++) {
        int n_samples = 0;
        float *audio = tts_vocoder_run(&voc, codes, n_steps, &n_samples, &timings[r]);
        if (!audio) {
            fprintf(stderr, "Error: vocoder run %d failed\n", r);
            free(codes);
            free(ref_audio);
            free(timings);
            free(last_audio);
            tts_vocoder_free(&voc);
            return -1;
        }

        printf("  Run %d: %.0f ms (%d samples)\n", r + 1, timings[r].total_ms, n_samples);

        /* Keep last run's audio for comparison and output */
        free(last_audio);
        last_audio = audio;
        last_n = n_samples;
    }

    /* Compute median timing */
    printf("\nVocoder benchmark: %d steps (%.1fs audio), %d runs (median)\n",
           n_steps, (double)n_steps * VOC_UPSAMPLE_TOTAL / VOC_SAMPLE_RATE, runs);

    if (runs == 1) {
        print_timing(&timings[0]);
    } else {
        /* Build median timing from per-field medians */
        voc_timing_t median_t;
        memset(&median_t, 0, sizeof(median_t));

        double *vals = (double *)malloc(runs * sizeof(double));

#define MEDIAN_FIELD(field) do { \
            for (int r = 0; r < runs; r++) vals[r] = timings[r].field; \
            median_t.field = median_of(vals, runs); \
        } while (0)

        MEDIAN_FIELD(rvq_ms);
        MEDIAN_FIELD(preconv_ms);
        MEDIAN_FIELD(xfmr_ms);
        for (int s = 0; s < VOC_UPSAMPLE_STAGES; s++) {
            for (int r = 0; r < runs; r++) vals[r] = timings[r].upsample_ms[s];
            median_t.upsample_ms[s] = median_of(vals, runs);
        }
        MEDIAN_FIELD(bigvgan_init_ms);
        for (int b = 0; b < VOC_BIGVGAN_NUM_BLOCKS; b++) {
            for (int r = 0; r < runs; r++) vals[r] = timings[r].bigvgan_block_ms[b];
            median_t.bigvgan_block_ms[b] = median_of(vals, runs);
            for (int r = 0; r < runs; r++) vals[r] = timings[r].bigvgan_tconv_ms[b];
            median_t.bigvgan_tconv_ms[b] = median_of(vals, runs);
            for (int r = 0; r < runs; r++) vals[r] = timings[r].bigvgan_res_ms[b];
            median_t.bigvgan_res_ms[b] = median_of(vals, runs);
        }
        MEDIAN_FIELD(final_ms);
        MEDIAN_FIELD(total_ms);

#undef MEDIAN_FIELD

        free(vals);
        print_timing(&median_t);
    }

    /* Save output if requested */
    if (out_path && last_audio) {
        save_raw(out_path, last_audio, (size_t)last_n * sizeof(float));
        printf("\nSaved output: %s (%d samples)\n", out_path, last_n);
    }

    /* Compare against reference */
    if (ref_audio && last_audio) {
        compare_result_t cmp;
        compare_audio(last_audio, last_n, ref_audio, ref_n, &cmp);

        printf("\nComparison vs reference:\n");
        printf("  Correlation: %f\n", cmp.correlation);
        if (isinf(cmp.snr_db)) {
            printf("  SNR: inf dB\n");
        } else {
            printf("  SNR: %.1f dB\n", cmp.snr_db);
        }
        printf("  Max diff: %f\n", cmp.max_diff);
        if (cmp.n_samples != cmp.ref_samples) {
            printf("  WARNING: sample count mismatch (test=%d, ref=%d)\n",
                   cmp.n_samples, cmp.ref_samples);
        }
    }

    free(codes);
    free(ref_audio);
    free(timings);
    free(last_audio);
    tts_vocoder_free(&voc);
    return 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */

static void print_usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s --model <dir> --codes <file.codes> [--ref <ref.raw>] [--runs N] [--out <output.raw>]\n", prog);
    printf("  %s --model <dir> --generate [--out-codes <file.codes>] [--out <file.raw>]\n", prog);
    printf("\nBenchmark mode:\n");
    printf("  --model <dir>     Qwen3-TTS-Tokenizer-12Hz directory (contains model.safetensors)\n");
    printf("  --codes <file>    Binary codec tokens file (int64, n_steps * 16)\n");
    printf("  --ref <file>      Reference audio for comparison (float32)\n");
    printf("  --runs <N>        Number of benchmark runs (default: 3)\n");
    printf("  --out <file>      Save vocoder output as raw float32\n");
    printf("\nGenerate mode (requires TTS pipeline):\n");
    printf("  --model <dir>     Qwen3-TTS model directory (parent of Tokenizer-12Hz)\n");
    printf("  --generate        Run full TTS pipeline with srand(42) to produce reference data\n");
    printf("  --out-codes <f>   Output codes path (default: bench_ref.codes)\n");
    printf("  --out <file>      Output audio path (default: bench_ref.raw)\n");
}

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
    const char *codes_path = NULL;
    const char *ref_path = NULL;
    const char *out_path = NULL;
    const char *out_codes_path = NULL;
    int runs = 3;
    int generate = 0;

    for (int i = 1; i < argc; i++) {
        const char *val;
        if ((val = parse_arg(argv[i], "--model")) != NULL) {
            model_dir = val;
        } else if ((val = parse_arg(argv[i], "--codes")) != NULL) {
            codes_path = val;
        } else if ((val = parse_arg(argv[i], "--ref")) != NULL) {
            ref_path = val;
        } else if ((val = parse_arg(argv[i], "--out")) != NULL) {
            out_path = val;
        } else if ((val = parse_arg(argv[i], "--out-codes")) != NULL) {
            out_codes_path = val;
        } else if ((val = parse_arg(argv[i], "--runs")) != NULL) {
            runs = atoi(val);
            if (runs < 1) runs = 1;
        } else if (strcmp(argv[i], "--generate") == 0) {
            generate = 1;
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
        fprintf(stderr, "Error: --model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (generate) {
#ifdef USE_ORT
        return do_generate(model_dir, out_codes_path, out_path);
#else
        fprintf(stderr, "Error: --generate requires ORT support (build with USE_ORT)\n");
        return 1;
#endif
    }

    if (!codes_path) {
        fprintf(stderr, "Error: --codes is required in benchmark mode\n\n");
        print_usage(argv[0]);
        return 1;
    }

    return do_benchmark(model_dir, codes_path, ref_path, out_path, runs);
}
