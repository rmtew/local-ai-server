/*
 * voice_presets.c - CLI tool for voice preset generation
 *
 * Standalone tool that replaces the Python voice preset pipeline:
 *   extract  - Pull clips from media via ffmpeg
 *   generate - Compute speaker embeddings and write voice_presets.bin
 *   list     - Dump existing preset file contents
 *
 * Usage:
 *   voice-presets extract --input movie.mp4 --name Chelsie \
 *       --timestamps "1:23-1:45, 3:10-3:22" --output-dir voice_samples/
 *
 *   voice-presets generate --model <tts-model-dir> --samples voice_samples/
 *
 *   voice-presets list --presets voice_presets.bin
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "platform.h"

#include "tts_mel.h"
#include "tts_speaker_enc.h"
#include "tts_voice_presets.h"
#include "tts_pipeline.h"
#include "qwen_asr_safetensors.h"
#include "qwen_asr_kernels.h"

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif

#ifdef _WIN32
#include <direct.h>
#define mkdir_p(path) _mkdir(path)
#else
#include <sys/stat.h>
#include <dirent.h>
#define mkdir_p(path) mkdir(path, 0755)
#endif

/* Required by tts_native.c */
volatile int g_shutdown = 0;

#define MAX_CLIPS       256
#define MAX_VOICES      64
#define EMBED_DIM       1024
#define COSINE_PRUNE_THRESHOLD  0.75f
#define STABILITY_THRESHOLD     0.05f

/* ========================================================================
 * WAV loading at 24kHz (for mel spectrogram input)
 *
 * Same chunk parsing as qwen_asr_audio.c but resamples to 24kHz.
 * ======================================================================== */

static uint16_t read_u16(const uint8_t *p) { return p[0] | (p[1] << 8); }
static uint32_t read_u32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24); }

/* Modified Bessel function I0 for Kaiser window */
static double bessel_i0(double x) {
    double sum = 1.0, term = 1.0, xx = x * x;
    for (int k = 1; k <= 20; k++) {
        term *= xx / (4.0 * (double)k * (double)k);
        sum += term;
    }
    return sum;
}

static float *load_wav_24k(const char *path, int *out_n_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open WAV file: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    if (file_size <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);

    uint8_t *data = (uint8_t *)malloc(file_size);
    if (!data || fread(data, 1, file_size, f) != (size_t)file_size) {
        fclose(f); free(data); return NULL;
    }
    fclose(f);

    /* Validate WAV header */
    if (file_size < 44 || memcmp(data, "RIFF", 4) != 0 || memcmp(data + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "Not a valid WAV file: %s\n", path);
        free(data);
        return NULL;
    }

    int channels = 0, sample_rate = 0, bits_per_sample = 0, audio_format = 0;
    const uint8_t *pcm_data = NULL;
    int pcm_size = 0;

    /* Parse chunks */
    const uint8_t *p = data + 12;
    const uint8_t *end = data + file_size;
    while (p + 8 <= end) {
        uint32_t chunk_size = read_u32(p + 4);
        if (p + 8 + chunk_size > end) break;
        if (memcmp(p, "fmt ", 4) == 0 && chunk_size >= 16) {
            audio_format = read_u16(p + 8);
            channels = read_u16(p + 10);
            sample_rate = read_u32(p + 12);
            bits_per_sample = read_u16(p + 22);
        } else if (memcmp(p, "data", 4) == 0) {
            pcm_data = p + 8;
            pcm_size = chunk_size;
            if (pcm_data + pcm_size > end) pcm_size = (int)(end - pcm_data);
        }
        p += 8 + chunk_size;
        if (chunk_size & 1) p++;
    }

    if (audio_format != 1 || bits_per_sample != 16 || pcm_data == NULL || channels < 1) {
        fprintf(stderr, "Unsupported WAV format: fmt=%d bits=%d channels=%d (%s)\n",
                audio_format, bits_per_sample, channels, path);
        free(data);
        return NULL;
    }

    /* Convert int16 PCM to float, mix channels to mono */
    int n_frames = pcm_size / (channels * 2);
    float *samples = (float *)malloc(n_frames * sizeof(float));
    if (!samples) { free(data); return NULL; }

    const int16_t *src = (const int16_t *)pcm_data;
    for (int i = 0; i < n_frames; i++) {
        if (channels == 1) {
            samples[i] = src[i] / 32768.0f;
        } else {
            float sum = 0;
            for (int c = 0; c < channels; c++) {
                int16_t val;
                memcpy(&val, &src[i * channels + c], sizeof(int16_t));
                sum += val;
            }
            samples[i] = (sum / channels) / 32768.0f;
        }
    }

    free(data);

    /* Resample to 24kHz if needed */
    if (sample_rate != TTS_MEL_SAMPLE_RATE) {
        int new_n = (int)((long long)n_frames * TTS_MEL_SAMPLE_RATE / sample_rate);
        float *resampled = (float *)malloc(new_n * sizeof(float));
        if (!resampled) { free(samples); return NULL; }

        const int SINC_HALF = 16;
        const double KAISER_BETA = 6.0;
        double ratio = (double)TTS_MEL_SAMPLE_RATE / (double)sample_rate;
        double cutoff = (ratio < 1.0) ? ratio : 1.0;
        double inv_I0_beta = 1.0 / bessel_i0(KAISER_BETA);

        for (int i = 0; i < new_n; i++) {
            double src_pos = (double)i / ratio;
            int center = (int)src_pos;
            double acc = 0.0, wsum = 0.0;

            int j_lo = center - SINC_HALF + 1;
            int j_hi = center + SINC_HALF;
            for (int j = j_lo; j <= j_hi; j++) {
                double d = (double)j - src_pos;
                double x = d * cutoff;
                double s = (fabs(x) < 1e-9) ? 1.0 : sin(M_PI * x) / (M_PI * x);

                double npos = d / SINC_HALF;
                double w = 0.0;
                if (npos > -1.0 && npos < 1.0) {
                    w = bessel_i0(KAISER_BETA * sqrt(1.0 - npos * npos)) * inv_I0_beta;
                }

                double coeff = s * w * cutoff;
                if (j >= 0 && j < n_frames) {
                    acc += samples[j] * coeff;
                }
                wsum += coeff;
            }
            resampled[i] = (wsum > 1e-9) ? (float)(acc / wsum) : 0.0f;
        }
        free(samples);
        samples = resampled;
        n_frames = new_n;
    }

    *out_n_samples = n_frames;
    return samples;
}

/* ========================================================================
 * Cosine similarity
 * ======================================================================== */

static float cosine_sim(const float *a, const float *b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

/* ========================================================================
 * Timestamp parsing
 *
 * Formats: "M:SS-M:SS", "H:MM:SS-H:MM:SS", "SS-SS"
 * Multiple timestamps separated by comma.
 * ======================================================================== */

typedef struct {
    double start;
    double end;
} timestamp_t;

/* Parse a single time value like "1:23", "1:02:30", or "85" into seconds. */
static double parse_time(const char *s, const char **end_ptr) {
    double parts[3] = {0, 0, 0};
    int n_parts = 0;
    const char *p = s;

    while (*p && n_parts < 3) {
        /* Skip whitespace */
        while (*p == ' ') p++;

        char *num_end;
        parts[n_parts] = strtod(p, &num_end);
        if (num_end == p) break;
        n_parts++;
        p = num_end;

        if (*p == ':') {
            p++;
        } else {
            break;
        }
    }

    if (end_ptr) *end_ptr = p;

    if (n_parts == 3) {
        /* H:MM:SS */
        return parts[0] * 3600 + parts[1] * 60 + parts[2];
    } else if (n_parts == 2) {
        /* M:SS */
        return parts[0] * 60 + parts[1];
    } else {
        /* SS */
        return parts[0];
    }
}

static int parse_timestamps(const char *input, timestamp_t *out, int max_ts) {
    int count = 0;
    const char *p = input;

    while (*p && count < max_ts) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;

        const char *after;
        double start = parse_time(p, &after);
        p = after;

        /* Skip separator '-' */
        while (*p == ' ') p++;
        if (*p != '-') {
            fprintf(stderr, "Expected '-' in timestamp range at: %s\n", p);
            return -1;
        }
        p++;
        while (*p == ' ') p++;

        double end = parse_time(p, &after);
        p = after;

        if (end <= start) {
            fprintf(stderr, "Invalid timestamp range: %.1f-%.1f (end must be > start)\n", start, end);
            return -1;
        }

        out[count].start = start;
        out[count].end = end;
        count++;
    }

    return count;
}

/* ========================================================================
 * Format time as string for ffmpeg -ss/-to
 * ======================================================================== */

static void format_time(double secs, char *buf, size_t buf_sz) {
    int h = (int)(secs / 3600);
    int m = (int)(fmod(secs, 3600) / 60);
    double s = fmod(secs, 60);
    if (h > 0) {
        snprintf(buf, buf_sz, "%d:%02d:%06.3f", h, m, s);
    } else {
        snprintf(buf, buf_sz, "%d:%06.3f", m, s);
    }
}

/* ========================================================================
 * Subcommand: extract
 * ======================================================================== */

static int find_next_clip_number(const char *dir) {
    /* Find existing clip_NNN.wav files and return next number */
    int max_num = 0;

#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s\\clip_*.wav", dir);
    HANDLE hFind = FindFirstFileA(pattern, &fd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            int n = 0;
            if (sscanf(fd.cFileName, "clip_%d.wav", &n) == 1 && n > max_num) {
                max_num = n;
            }
        } while (FindNextFileA(hFind, &fd));
        FindClose(hFind);
    }
#else
    /* Fallback: try clip_001 through clip_999 */
    for (int i = 1; i <= 999; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/clip_%03d.wav", dir, i);
        FILE *f = fopen(path, "rb");
        if (f) {
            fclose(f);
            if (i > max_num) max_num = i;
        }
    }
#endif

    return max_num + 1;
}

static int do_extract(const char *input, const char *name,
                      const char *timestamps_str, const char *output_dir,
                      int append) {
    /* Parse timestamps */
    timestamp_t timestamps[MAX_CLIPS];
    int n_ts = parse_timestamps(timestamps_str, timestamps, MAX_CLIPS);
    if (n_ts <= 0) {
        fprintf(stderr, "No valid timestamps found\n");
        return 1;
    }

    printf("Extracting %d clip%s from: %s\n", n_ts, n_ts > 1 ? "s" : "", input);

    /* Create output directory: <output_dir>/<name>/ */
    mkdir_p(output_dir);
    char voice_dir[512];
    snprintf(voice_dir, sizeof(voice_dir), "%s/%s", output_dir, name);
    mkdir_p(voice_dir);

    /* Determine starting clip number */
    int clip_num = append ? find_next_clip_number(voice_dir) : 1;

    for (int i = 0; i < n_ts; i++) {
        char start_str[32], end_str[32];
        format_time(timestamps[i].start, start_str, sizeof(start_str));
        format_time(timestamps[i].end, end_str, sizeof(end_str));

        char out_path[512];
        snprintf(out_path, sizeof(out_path), "%s/clip_%03d.wav", voice_dir, clip_num);

        char cmd[1024];
        snprintf(cmd, sizeof(cmd),
                 "ffmpeg -y -i \"%s\" -ss %s -to %s -ar 24000 -ac 1 -sample_fmt s16 \"%s\" 2>&1",
                 input, start_str, end_str, out_path);

        printf("  [%d/%d] %s -> %s ... ", i + 1, n_ts, start_str, end_str);
        fflush(stdout);

        int rc = system(cmd);
        if (rc != 0) {
            printf("FAILED (ffmpeg exit code %d)\n", rc);
            fprintf(stderr, "  Command: %s\n", cmd);
            return 1;
        }

        /* Verify output exists */
        FILE *check = fopen(out_path, "rb");
        if (!check) {
            printf("FAILED (output file not created)\n");
            return 1;
        }
        fseek(check, 0, SEEK_END);
        long sz = ftell(check);
        fclose(check);
        printf("OK (%.1f KB)\n", sz / 1024.0);

        clip_num++;
    }

    printf("\nExtracted %d clips to %s/\n", n_ts, voice_dir);
    return 0;
}

/* ========================================================================
 * Subcommand: generate
 * ======================================================================== */

typedef struct {
    char name[TTS_PRESET_NAME_LEN];
    float embeddings[MAX_CLIPS][EMBED_DIM];
    int n_clips;
    int used[MAX_CLIPS];        /* 1 = kept, 0 = pruned */
    float avg_embed[EMBED_DIM]; /* final averaged embedding */
    float stability_score;      /* leave-one-out stability */
    float roundtrip_sim;        /* cosine sim after round-trip verification */
} voice_data_t;

/* Compute centroid of kept embeddings */
static void compute_centroid(voice_data_t *v) {
    memset(v->avg_embed, 0, sizeof(v->avg_embed));
    int count = 0;
    for (int i = 0; i < v->n_clips; i++) {
        if (!v->used[i]) continue;
        for (int d = 0; d < EMBED_DIM; d++) {
            v->avg_embed[d] += v->embeddings[i][d];
        }
        count++;
    }
    if (count > 0) {
        for (int d = 0; d < EMBED_DIM; d++) {
            v->avg_embed[d] /= count;
        }
    }
}

/* Prune outlier clips: drop any with cosine sim to centroid below threshold */
static int prune_outliers(voice_data_t *v) {
    /* Mark all clips as used first */
    for (int i = 0; i < v->n_clips; i++) v->used[i] = 1;

    if (v->n_clips <= 1) {
        compute_centroid(v);
        return 0;
    }

    /* First pass: compute centroid of all clips */
    compute_centroid(v);

    /* Second pass: prune outliers */
    int pruned = 0;
    for (int i = 0; i < v->n_clips; i++) {
        float sim = cosine_sim(v->embeddings[i], v->avg_embed, EMBED_DIM);
        if (sim < COSINE_PRUNE_THRESHOLD) {
            v->used[i] = 0;
            pruned++;
            printf("    clip %d: pruned (sim=%.3f < %.2f)\n",
                   i + 1, sim, COSINE_PRUNE_THRESHOLD);
        }
    }

    /* Recompute centroid with remaining clips */
    if (pruned > 0) {
        compute_centroid(v);
    }

    return pruned;
}

/* Leave-one-out stability: how much does removing any single clip change the average? */
static float check_convergence(voice_data_t *v) {
    int kept = 0;
    for (int i = 0; i < v->n_clips; i++) {
        if (v->used[i]) kept++;
    }
    if (kept <= 2) return 1.0f;  /* can't do LOO with < 3 clips */

    float max_change = 0.0f;

    for (int leave = 0; leave < v->n_clips; leave++) {
        if (!v->used[leave]) continue;

        /* Compute centroid without this clip */
        float loo_embed[EMBED_DIM];
        memset(loo_embed, 0, sizeof(loo_embed));
        int count = 0;
        for (int i = 0; i < v->n_clips; i++) {
            if (!v->used[i] || i == leave) continue;
            for (int d = 0; d < EMBED_DIM; d++) {
                loo_embed[d] += v->embeddings[i][d];
            }
            count++;
        }
        for (int d = 0; d < EMBED_DIM; d++) {
            loo_embed[d] /= count;
        }

        /* Measure change from full centroid */
        float sim = cosine_sim(loo_embed, v->avg_embed, EMBED_DIM);
        float change = 1.0f - sim;
        if (change > max_change) max_change = change;
    }

    return max_change;
}

/* Write voice_presets.bin */
static int write_presets_bin(const char *path, voice_data_t *voices, int n_voices) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot write to %s\n", path);
        return -1;
    }

    int32_t n = n_voices;
    fwrite(&n, sizeof(int32_t), 1, f);

    for (int i = 0; i < n_voices; i++) {
        /* Name: 64 bytes null-padded */
        char name_buf[TTS_PRESET_NAME_LEN];
        memset(name_buf, 0, sizeof(name_buf));
        strncpy(name_buf, voices[i].name, TTS_PRESET_NAME_LEN - 1);
        fwrite(name_buf, 1, TTS_PRESET_NAME_LEN, f);

        /* Embedding: 1024 floats */
        fwrite(voices[i].avg_embed, sizeof(float), EMBED_DIM, f);
    }

    fclose(f);
    return 0;
}

static int do_generate(const char *model_dir, const char *samples_dir,
                       const char *output_path, int do_roundtrip, int verbose) {
    printf("=== Voice Preset Generation ===\n");
    printf("Model: %s\n", model_dir);
    printf("Samples: %s\n", samples_dir);

    /* Scan samples directory for voice subdirectories.
     * voice_data_t is large (~1MB each due to embedding arrays), so heap-allocate. */
    voice_data_t *voices = (voice_data_t *)calloc(MAX_VOICES, sizeof(voice_data_t));
    if (!voices) {
        fprintf(stderr, "Failed to allocate voice data\n");
        return 1;
    }
    int n_voices = 0;

#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    char search_path[512];
    snprintf(search_path, sizeof(search_path), "%s\\*", samples_dir);
    HANDLE hFind = FindFirstFileA(search_path, &fd);
    if (hFind == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Cannot open samples directory: %s\n", samples_dir);
        free(voices);
        return 1;
    }
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
        if (fd.cFileName[0] == '.') continue;
        if (n_voices >= MAX_VOICES) break;

        memset(&voices[n_voices], 0, sizeof(voice_data_t));
        strncpy(voices[n_voices].name, fd.cFileName, TTS_PRESET_NAME_LEN - 1);
        n_voices++;
    } while (FindNextFileA(hFind, &fd));
    FindClose(hFind);
#else
    /* POSIX: use opendir/readdir */
    DIR *dir = opendir(samples_dir);
    if (!dir) {
        fprintf(stderr, "Cannot open samples directory: %s\n", samples_dir);
        free(voices);
        return 1;
    }
    struct dirent *de;
    while ((de = readdir(dir)) != NULL && n_voices < MAX_VOICES) {
        if (de->d_name[0] == '.') continue;
        /* Check if it's a directory */
        char subpath[512];
        snprintf(subpath, sizeof(subpath), "%s/%s", samples_dir, de->d_name);
        struct stat st;
        if (stat(subpath, &st) == 0 && S_ISDIR(st.st_mode)) {
            memset(&voices[n_voices], 0, sizeof(voice_data_t));
            strncpy(voices[n_voices].name, de->d_name, TTS_PRESET_NAME_LEN - 1);
            n_voices++;
        }
    }
    closedir(dir);
#endif

    if (n_voices == 0) {
        fprintf(stderr, "No voice subdirectories found in %s\n", samples_dir);
        fprintf(stderr, "Expected structure: %s/<voice_name>/clip_001.wav ...\n", samples_dir);
        free(voices);
        return 1;
    }

    printf("Found %d voice%s\n\n", n_voices, n_voices > 1 ? "s" : "");

    /* Load speaker encoder weights (lightweight — only ~17MB) */
    printf("Loading speaker encoder from %s...\n", model_dir);
    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "Failed to open safetensors from %s\n", model_dir);
        free(voices);
        return 1;
    }

    tts_speaker_enc_ctx_t enc;
    if (tts_speaker_enc_init(&enc, ms, verbose) != 0) {
        fprintf(stderr, "Failed to init speaker encoder (not a Base model?)\n");
        multi_safetensors_close(ms);
        free(voices);
        return 1;
    }
    printf("Speaker encoder loaded OK\n\n");

    /* Process each voice */
    for (int v = 0; v < n_voices; v++) {
        printf("--- Voice: %s ---\n", voices[v].name);

        /* Find WAV files in voice subdirectory */
        char voice_path[512];
        snprintf(voice_path, sizeof(voice_path), "%s/%s", samples_dir, voices[v].name);

#ifdef _WIN32
        char wav_pattern[512];
        snprintf(wav_pattern, sizeof(wav_pattern), "%s\\*.wav", voice_path);
        HANDLE hWav = FindFirstFileA(wav_pattern, &fd);
        if (hWav == INVALID_HANDLE_VALUE) {
            printf("  No WAV files found, skipping\n\n");
            continue;
        }
        do {
            if (voices[v].n_clips >= MAX_CLIPS) break;
            char wav_path[512];
            snprintf(wav_path, sizeof(wav_path), "%s/%s", voice_path, fd.cFileName);
#else
        DIR *vdir = opendir(voice_path);
        if (!vdir) {
            printf("  Cannot open directory, skipping\n\n");
            continue;
        }
        struct dirent *vde;
        while ((vde = readdir(vdir)) != NULL) {
            if (voices[v].n_clips >= MAX_CLIPS) break;
            size_t namelen = strlen(vde->d_name);
            if (namelen < 5) continue;
            if (strcasecmp(vde->d_name + namelen - 4, ".wav") != 0) continue;
            char wav_path[512];
            snprintf(wav_path, sizeof(wav_path), "%s/%s", voice_path, vde->d_name);
#endif
            int clip_idx = voices[v].n_clips;

            /* Load WAV at 24kHz */
            int n_samples = 0;
            float *audio = load_wav_24k(wav_path, &n_samples);
            if (!audio || n_samples == 0) {
                printf("  [clip %d] FAILED to load %s\n", clip_idx + 1, wav_path);
                free(audio);
#ifdef _WIN32
                continue;
#else
                continue;
#endif
            }

            double duration = (double)n_samples / TTS_MEL_SAMPLE_RATE;
            if (verbose) {
                printf("  [clip %d] %s (%.1fs, %d samples)\n",
                       clip_idx + 1, wav_path, duration, n_samples);
            }

            /* Compute mel spectrogram */
            int n_frames = 0;
            float *mel = tts_mel_compute(audio, n_samples, &n_frames);
            free(audio);

            if (!mel || n_frames == 0) {
                printf("  [clip %d] mel computation failed\n", clip_idx + 1);
                free(mel);
                continue;
            }

            if (verbose) {
                printf("    mel: [128, %d]\n", n_frames);
            }

            /* Run speaker encoder */
            int rc = tts_speaker_enc_forward(&enc, mel, n_frames,
                                              voices[v].embeddings[clip_idx]);
            free(mel);

            if (rc != 0) {
                printf("  [clip %d] speaker encoder failed\n", clip_idx + 1);
                continue;
            }

            /* Compute embedding norm for diagnostics */
            float norm = 0;
            for (int d = 0; d < EMBED_DIM; d++) {
                norm += voices[v].embeddings[clip_idx][d] * voices[v].embeddings[clip_idx][d];
            }
            norm = sqrtf(norm);

            printf("  [clip %d] OK  norm=%.4f", clip_idx + 1, norm);
            if (verbose) {
                float mn = voices[v].embeddings[clip_idx][0];
                float mx = mn;
                for (int d = 1; d < EMBED_DIM; d++) {
                    float val = voices[v].embeddings[clip_idx][d];
                    if (val < mn) mn = val;
                    if (val > mx) mx = val;
                }
                printf("  range=[%.4f, %.4f]", mn, mx);
            }
            printf("\n");

            voices[v].n_clips++;
#ifdef _WIN32
        } while (FindNextFileA(hWav, &fd));
        FindClose(hWav);
#else
        }
        closedir(vdir);
#endif

        if (voices[v].n_clips == 0) {
            printf("  No valid clips processed\n\n");
            continue;
        }

        /* Prune outliers */
        int pruned = prune_outliers(&voices[v]);

        int kept = 0;
        for (int i = 0; i < voices[v].n_clips; i++) {
            if (voices[v].used[i]) kept++;
        }

        if (kept == 0) {
            printf("  All clips pruned! Skipping this voice.\n\n");
            continue;
        }

        printf("  Clips: %d used, %d pruned\n", kept, pruned);

        /* Check convergence (leave-one-out stability) */
        voices[v].stability_score = check_convergence(&voices[v]);
        printf("  Stability: %.4f", voices[v].stability_score);
        if (voices[v].stability_score > STABILITY_THRESHOLD) {
            printf(" (WARNING: >%.0f%% — consider adding more clips)", STABILITY_THRESHOLD * 100);
        } else {
            printf(" (good)");
        }
        printf("\n");

        /* Print pairwise similarities */
        if (kept > 1 && verbose) {
            printf("  Pairwise cosine similarities:\n");
            for (int i = 0; i < voices[v].n_clips; i++) {
                if (!voices[v].used[i]) continue;
                for (int j = i + 1; j < voices[v].n_clips; j++) {
                    if (!voices[v].used[j]) continue;
                    float sim = cosine_sim(voices[v].embeddings[i],
                                           voices[v].embeddings[j], EMBED_DIM);
                    printf("    clip %d vs %d: %.4f\n", i + 1, j + 1, sim);
                }
            }
        }

        printf("\n");
    }

    /* Release speaker encoder (no longer needed) */
    tts_speaker_enc_free(&enc);
    multi_safetensors_close(ms);

    /* Filter out voices with no valid clips */
    int valid_voices = 0;
    for (int v = 0; v < n_voices; v++) {
        int kept = 0;
        for (int i = 0; i < voices[v].n_clips; i++) {
            if (voices[v].used[i]) kept++;
        }
        if (kept > 0) {
            if (valid_voices != v) {
                voices[valid_voices] = voices[v];
            }
            valid_voices++;
        }
    }
    n_voices = valid_voices;

    if (n_voices == 0) {
        fprintf(stderr, "No valid voices to write\n");
        free(voices);
        return 1;
    }

    /* Determine output path */
    char preset_path[512];
    if (output_path) {
        snprintf(preset_path, sizeof(preset_path), "%s", output_path);
    } else {
        snprintf(preset_path, sizeof(preset_path), "%s/voice_presets.bin", model_dir);
    }

    /* Write presets (before round-trip, so we have a file to load) */
    if (write_presets_bin(preset_path, voices, n_voices) != 0) {
        free(voices);
        return 1;
    }
    printf("Wrote %d preset%s to %s\n", n_voices, n_voices > 1 ? "s" : "", preset_path);

    /* Round-trip verification */
    if (do_roundtrip) {
        printf("\n=== Round-trip Verification ===\n");
        printf("Loading full TTS pipeline...\n");

        qwen_set_threads(4);

        TtsPipeline tts;
        if (tts_pipeline_init(&tts, model_dir, verbose) != 0) {
            fprintf(stderr, "WARNING: Failed to init TTS pipeline, skipping round-trip\n");
        } else {
            const char *test_sentence = "The quick brown fox jumps over the lazy dog.";

            for (int v = 0; v < n_voices; v++) {
                printf("  Voice '%s': ", voices[v].name);
                fflush(stdout);

                /* Synthesize with this voice preset */
                srand(42);
                TtsResult result;
                int rc = tts_pipeline_synthesize(&tts, test_sentence,
                                                  voices[v].name, NULL,
                                                  0.3f, 50, 1.0f, &result);
                if (rc != 0 || !result.wav_data) {
                    printf("synthesis FAILED\n");
                    continue;
                }

                /* Extract float audio from WAV for mel computation.
                 * WAV has 44-byte header, then int16 PCM. */
                int n_pcm = result.n_samples;
                float *synth_audio = (float *)malloc(n_pcm * sizeof(float));
                if (synth_audio) {
                    const int16_t *pcm = (const int16_t *)(result.wav_data + 44);
                    for (int i = 0; i < n_pcm; i++) {
                        synth_audio[i] = pcm[i] / 32768.0f;
                    }

                    /* Compute mel + speaker embedding on synthesized audio */
                    int n_frames = 0;
                    float *mel = tts_mel_compute(synth_audio, n_pcm, &n_frames);
                    free(synth_audio);

                    if (mel && n_frames > 0) {
                        /* Need speaker encoder again for round-trip.
                         * The pipeline has one internally. */
                        float rt_embed[EMBED_DIM];
                        int enc_rc = tts_speaker_enc_forward(
                            &tts.native->speaker_enc, mel, n_frames, rt_embed);
                        free(mel);

                        if (enc_rc == 0) {
                            float sim = cosine_sim(voices[v].avg_embed, rt_embed, EMBED_DIM);
                            voices[v].roundtrip_sim = sim;

                            const char *quality;
                            if (sim > 0.85f) quality = "GOOD";
                            else if (sim > 0.70f) quality = "FAIR";
                            else quality = "POOR";

                            printf("similarity=%.4f (%s)\n", sim, quality);
                        } else {
                            printf("speaker encoder failed on synth audio\n");
                        }
                    } else {
                        free(mel);
                        printf("mel failed on synth audio\n");
                    }
                }

                free(result.wav_data);
            }

            tts_pipeline_free(&tts);
        }
    }

    /* Final report */
    printf("\n=== Summary ===\n");
    printf("%-20s  %5s  %5s  %8s  %8s", "Voice", "Clips", "Used", "Stability", "Roundtrip");
    printf("\n");
    printf("%-20s  %5s  %5s  %8s  %8s", "-----", "-----", "----", "---------", "---------");
    printf("\n");

    for (int v = 0; v < n_voices; v++) {
        int kept = 0;
        for (int i = 0; i < voices[v].n_clips; i++) {
            if (voices[v].used[i]) kept++;
        }
        printf("%-20s  %5d  %5d  %8.4f", voices[v].name, voices[v].n_clips, kept,
               voices[v].stability_score);
        if (voices[v].roundtrip_sim > 0) {
            printf("  %8.4f", voices[v].roundtrip_sim);
        } else {
            printf("  %8s", "-");
        }
        printf("\n");
    }

    printf("\nOutput: %s\n", preset_path);
    free(voices);
    return 0;
}

/* ========================================================================
 * Subcommand: list
 * ======================================================================== */

static int do_list(const char *preset_path) {
    tts_voice_presets_t vp;
    if (tts_voice_presets_load(&vp, preset_path) != 0) {
        fprintf(stderr, "Cannot load presets from %s\n", preset_path);
        return 1;
    }

    printf("%d preset%s in %s:\n\n", vp.n_presets, vp.n_presets > 1 ? "s" : "", preset_path);
    printf("%-20s  %10s  %24s\n", "Name", "Norm", "Range");
    printf("%-20s  %10s  %24s\n", "----", "----", "-----");

    for (int i = 0; i < vp.n_presets; i++) {
        float *e = vp.presets[i].embed;
        float norm = 0, mn = e[0], mx = e[0];
        for (int d = 0; d < EMBED_DIM; d++) {
            norm += e[d] * e[d];
            if (e[d] < mn) mn = e[d];
            if (e[d] > mx) mx = e[d];
        }
        norm = sqrtf(norm);
        printf("%-20s  %10.4f  [%10.4f, %10.4f]\n",
               vp.presets[i].name, norm, mn, mx);
    }

    tts_voice_presets_free(&vp);
    return 0;
}

/* ========================================================================
 * Argument parsing and main
 * ======================================================================== */

static void print_usage(const char *prog) {
    printf("Voice preset generation tool for Qwen3-TTS voice cloning.\n\n");
    printf("Usage:\n");
    printf("  %s extract --input <media> --name <voice> --timestamps <ranges>\n", prog);
    printf("             [--output-dir <dir>] [--append]\n");
    printf("  %s generate --model <tts-model-dir> --samples <dir>\n", prog);
    printf("             [--output <path>] [--roundtrip] [--verbose]\n");
    printf("  %s list --presets <path>\n\n", prog);

    printf("Subcommands:\n");
    printf("  extract    Pull clips from media files via ffmpeg\n");
    printf("  generate   Compute speaker embeddings and write voice_presets.bin\n");
    printf("  list       Dump contents of an existing voice_presets.bin\n\n");

    printf("Extract options:\n");
    printf("  --input <file>        Input media file (any format ffmpeg supports)\n");
    printf("  --name <name>         Voice/speaker name\n");
    printf("  --timestamps <ranges> Timestamp ranges: \"M:SS-M:SS, M:SS-M:SS, ...\"\n");
    printf("  --output-dir <dir>    Output directory (default: voice_samples)\n");
    printf("  --append              Continue numbering from existing clips\n\n");

    printf("Generate options:\n");
    printf("  --model <dir>         TTS model directory (must contain model.safetensors)\n");
    printf("  --samples <dir>       Samples directory (one subfolder per voice with WAVs)\n");
    printf("  --output <path>       Output path (default: <model-dir>/voice_presets.bin)\n");
    printf("  --roundtrip           Verify by synthesizing and comparing embeddings\n");
    printf("  --verbose             Print detailed diagnostics\n\n");

    printf("List options:\n");
    printf("  --presets <path>      Path to voice_presets.bin\n\n");

    printf("Examples:\n");
    printf("  %s extract --input movie.mp4 --name Chelsie \\\n", prog);
    printf("      --timestamps \"1:23-1:45, 3:10-3:22\" --output-dir voice_samples/\n");
    printf("  %s generate --model /path/to/qwen3-tts-0.6b-base --samples voice_samples/\n", prog);
    printf("  %s list --presets /path/to/voice_presets.bin\n", prog);
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

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *subcmd = argv[1];

    /* ---- extract ---- */
    if (strcmp(subcmd, "extract") == 0) {
        const char *input = NULL;
        const char *name = NULL;
        const char *timestamps = NULL;
        const char *output_dir = "voice_samples";
        int append = 0;

        for (int i = 2; i < argc; i++) {
            const char *val;
            if ((val = parse_arg(argv[i], "--input")) != NULL) {
                input = val;
            } else if ((val = parse_arg(argv[i], "--name")) != NULL) {
                name = val;
            } else if ((val = parse_arg(argv[i], "--timestamps")) != NULL) {
                timestamps = val;
            } else if ((val = parse_arg(argv[i], "--output-dir")) != NULL) {
                output_dir = val;
            } else if (strcmp(argv[i], "--append") == 0) {
                append = 1;
            } else {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                return 1;
            }
        }

        if (!input || !name || !timestamps) {
            fprintf(stderr, "extract requires --input, --name, and --timestamps\n");
            return 1;
        }

        return do_extract(input, name, timestamps, output_dir, append);
    }

    /* ---- generate ---- */
    if (strcmp(subcmd, "generate") == 0) {
        const char *model_dir = NULL;
        const char *samples_dir = NULL;
        const char *output = NULL;
        int roundtrip = 0;
        int verbose = 0;

        for (int i = 2; i < argc; i++) {
            const char *val;
            if ((val = parse_arg(argv[i], "--model")) != NULL) {
                model_dir = val;
            } else if ((val = parse_arg(argv[i], "--samples")) != NULL) {
                samples_dir = val;
            } else if ((val = parse_arg(argv[i], "--output")) != NULL) {
                output = val;
            } else if (strcmp(argv[i], "--roundtrip") == 0) {
                roundtrip = 1;
            } else if (strcmp(argv[i], "--verbose") == 0) {
                verbose = 1;
            } else {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                return 1;
            }
        }

        if (!model_dir || !samples_dir) {
            fprintf(stderr, "generate requires --model and --samples\n");
            return 1;
        }

        return do_generate(model_dir, samples_dir, output, roundtrip, verbose);
    }

    /* ---- list ---- */
    if (strcmp(subcmd, "list") == 0) {
        const char *presets_path = NULL;

        for (int i = 2; i < argc; i++) {
            const char *val;
            if ((val = parse_arg(argv[i], "--presets")) != NULL) {
                presets_path = val;
            } else {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                return 1;
            }
        }

        if (!presets_path) {
            fprintf(stderr, "list requires --presets\n");
            return 1;
        }

        return do_list(presets_path);
    }

    /* Unknown subcommand */
    if (strcmp(subcmd, "--help") == 0 || strcmp(subcmd, "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    fprintf(stderr, "Unknown subcommand: %s\n", subcmd);
    print_usage(argv[0]);
    return 1;
}
