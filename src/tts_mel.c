/*
 * tts_mel.c - 128-band mel spectrogram for speaker encoder
 *
 * Implements STFT + slaney-norm mel filterbank + log compression.
 * Matches torchaudio.transforms.MelSpectrogram with slaney norm.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "tts_mel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Hann window (lazy-init) ---- */

static float g_hann_window[TTS_MEL_WIN];
static int g_hann_inited = 0;

static void ensure_hann_window(void) {
    if (g_hann_inited) return;
    for (int i = 0; i < TTS_MEL_WIN; i++) {
        g_hann_window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / TTS_MEL_WIN));
    }
    g_hann_inited = 1;
}

/* ---- Slaney mel filterbank ---- */

static float hz_to_mel(float hz) {
    /* Slaney / HTK-style: linear below 1000 Hz, log above */
    if (hz < 1000.0f) {
        return hz * 3.0f / 200.0f;  /* 15.0 at 1000 Hz */
    }
    return 15.0f + 27.0f * logf(hz / 1000.0f) / logf(6.4f);
}

static float mel_to_hz(float mel) {
    if (mel < 15.0f) {
        return mel * 200.0f / 3.0f;
    }
    return 1000.0f * expf((mel - 15.0f) * logf(6.4f) / 27.0f);
}

/* Build [n_mels, n_fft/2+1] filterbank with slaney normalization.
 * Returns malloc'd array. */
static float *build_mel_filterbank(void) {
    int n_mels = TTS_MEL_N_MELS;
    int n_bins = TTS_MEL_N_BINS;  /* 513 */
    float fmin = TTS_MEL_FMIN;
    float fmax = TTS_MEL_FMAX;
    int sr = TTS_MEL_SAMPLE_RATE;

    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);

    /* n_mels + 2 edges */
    int n_edges = n_mels + 2;
    float *mel_edges = (float *)malloc((size_t)n_edges * sizeof(float));
    float *hz_edges = (float *)malloc((size_t)n_edges * sizeof(float));
    if (!mel_edges || !hz_edges) { free(mel_edges); free(hz_edges); return NULL; }

    for (int i = 0; i < n_edges; i++) {
        mel_edges[i] = mel_min + (mel_max - mel_min) * i / (n_edges - 1);
        hz_edges[i] = mel_to_hz(mel_edges[i]);
    }

    /* Frequency bins in Hz */
    float *fft_freqs = (float *)malloc((size_t)n_bins * sizeof(float));
    if (!fft_freqs) { free(mel_edges); free(hz_edges); return NULL; }
    for (int i = 0; i < n_bins; i++) {
        fft_freqs[i] = (float)sr * i / TTS_MEL_N_FFT;
    }

    /* Build filterbank */
    float *fb = (float *)calloc((size_t)n_mels * n_bins, sizeof(float));
    if (!fb) { free(mel_edges); free(hz_edges); free(fft_freqs); return NULL; }

    for (int m = 0; m < n_mels; m++) {
        float left = hz_edges[m];
        float center = hz_edges[m + 1];
        float right = hz_edges[m + 2];

        /* Slaney normalization: 2 / (right - left) */
        float norm = 2.0f / (right - left);

        for (int k = 0; k < n_bins; k++) {
            float f = fft_freqs[k];
            float val = 0.0f;
            if (f >= left && f <= center && center > left) {
                val = (f - left) / (center - left);
            } else if (f > center && f <= right && right > center) {
                val = (right - f) / (right - center);
            }
            fb[m * n_bins + k] = val * norm;
        }
    }

    free(mel_edges);
    free(hz_edges);
    free(fft_freqs);
    return fb;
}

/* ---- Radix-2 FFT (Cooley-Tukey, in-place) ---- */

/* Bit-reversal permutation for N elements */
static void bit_reverse(float *re, float *im, int N) {
    int bits = 0;
    { int n = N; while (n > 1) { bits++; n >>= 1; } }

    for (int i = 0; i < N; i++) {
        int j = 0;
        for (int b = 0; b < bits; b++) {
            j |= ((i >> b) & 1) << (bits - 1 - b);
        }
        if (j > i) {
            float tmp;
            tmp = re[i]; re[i] = re[j]; re[j] = tmp;
            tmp = im[i]; im[i] = im[j]; im[j] = tmp;
        }
    }
}

/* In-place radix-2 DIT FFT */
static void fft_inplace(float *re, float *im, int N) {
    bit_reverse(re, im, N);

    for (int s = 2; s <= N; s <<= 1) {
        int half = s >> 1;
        float angle = -2.0f * (float)M_PI / s;
        float wn_re = cosf(angle);
        float wn_im = sinf(angle);

        for (int k = 0; k < N; k += s) {
            float w_re = 1.0f, w_im = 0.0f;
            for (int j = 0; j < half; j++) {
                int u = k + j;
                int v = u + half;
                float t_re = w_re * re[v] - w_im * im[v];
                float t_im = w_re * im[v] + w_im * re[v];
                re[v] = re[u] - t_re;
                im[v] = im[u] - t_im;
                re[u] += t_re;
                im[u] += t_im;
                float tmp = w_re * wn_re - w_im * wn_im;
                w_im = w_re * wn_im + w_im * wn_re;
                w_re = tmp;
            }
        }
    }
}

/* ---- Main mel spectrogram computation ---- */

float *tts_mel_compute(const float *audio, int n_samples, int *out_n_frames) {
    if (!audio || n_samples <= 0) return NULL;

    ensure_hann_window();

    int pad = TTS_MEL_PAD;
    int n_fft = TTS_MEL_N_FFT;
    int hop = TTS_MEL_HOP;
    int n_mels = TTS_MEL_N_MELS;
    int n_bins = TTS_MEL_N_BINS;

    /* Reflect-pad the audio */
    int padded_len = n_samples + 2 * pad;
    float *padded = (float *)malloc((size_t)padded_len * sizeof(float));
    if (!padded) return NULL;

    /* Left reflect padding */
    for (int i = 0; i < pad; i++) {
        int src = pad - i;
        if (src >= n_samples) src = n_samples - 1;
        padded[i] = audio[src];
    }
    /* Copy original */
    memcpy(padded + pad, audio, (size_t)n_samples * sizeof(float));
    /* Right reflect padding */
    for (int i = 0; i < pad; i++) {
        int src = n_samples - 2 - i;
        if (src < 0) src = 0;
        padded[pad + n_samples + i] = audio[src];
    }

    /* Number of frames */
    int n_frames = (padded_len - n_fft) / hop + 1;
    if (n_frames <= 0) { free(padded); return NULL; }

    /* Build mel filterbank */
    float *fb = build_mel_filterbank();
    if (!fb) { free(padded); return NULL; }

    /* Allocate STFT magnitude buffer for one frame */
    float *fft_re = (float *)malloc((size_t)n_fft * sizeof(float));
    float *fft_im = (float *)malloc((size_t)n_fft * sizeof(float));
    float *mag = (float *)malloc((size_t)n_bins * sizeof(float));

    /* Allocate output mel buffer */
    float *mel = (float *)calloc((size_t)n_mels * n_frames, sizeof(float));
    if (!fft_re || !fft_im || !mag || !mel) {
        free(padded); free(fb); free(fft_re); free(fft_im); free(mag); free(mel);
        return NULL;
    }

    /* Process each frame */
    for (int t = 0; t < n_frames; t++) {
        int offset = t * hop;

        /* Window the frame */
        for (int i = 0; i < n_fft; i++) {
            fft_re[i] = padded[offset + i] * g_hann_window[i];
            fft_im[i] = 0.0f;
        }

        /* FFT */
        fft_inplace(fft_re, fft_im, n_fft);

        /* Magnitude: sqrt(re^2 + im^2 + 1e-9) */
        for (int k = 0; k < n_bins; k++) {
            mag[k] = sqrtf(fft_re[k] * fft_re[k] + fft_im[k] * fft_im[k] + 1e-9f);
        }

        /* Mel projection: fb[n_mels, n_bins] @ mag[n_bins] */
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            const float *row = fb + m * n_bins;
            for (int k = 0; k < n_bins; k++) {
                sum += row[k] * mag[k];
            }
            /* Log compression: log(max(x, 1e-5)) */
            if (sum < 1e-5f) sum = 1e-5f;
            mel[m * n_frames + t] = logf(sum);
        }
    }

    free(padded);
    free(fb);
    free(fft_re);
    free(fft_im);
    free(mag);

    *out_n_frames = n_frames;
    return mel;
}
